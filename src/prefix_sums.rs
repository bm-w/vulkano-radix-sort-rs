use std::sync::Arc;

mod vk {
	pub(super) use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
	pub(super) use vulkano::command_buffer::AutoCommandBufferBuilder;
	pub(super) use vulkano::descriptor_set::{
		DescriptorSet, WriteDescriptorSet, allocator::DescriptorSetAllocator,
	};
	pub(super) use vulkano::memory::allocator::{
		AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
	};
	pub(super) use vulkano::pipeline::{
		ComputePipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
		compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
	};
	pub(super) use vulkano::shader::SpecializationConstant;
	pub(super) use vulkano::{Validated, VulkanError as Error};
}
use vulkano::{device::DeviceOwned as _, pipeline::Pipeline as _};

use crate::util::ParallelReduceExt;

/// A set of compute kernels that performs an in-place prefix sums operations
/// on an array of 32-bit unsigned integers.
pub struct PrefixSums {
	descriptor_set_allocator: Arc<dyn vk::DescriptorSetAllocator>,
	buffer_memory_allocator: Arc<vk::StandardMemoryAllocator>,
	local_pipeline: Arc<vk::ComputePipeline>,
	global_pipeline: Arc<vk::ComputePipeline>,
	work_group_size: u32,
}

/// The error type that can be returned by [`PrefixSums`]’s methods
#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("The auxiliary buffer has invalid length")]
	InvalidAuxiliaryLength,
	#[error(transparent)]
	Vulkan(#[from] vk::Error),
}

impl PrefixSums {
	/// Constructs a new instance of [`PrefixSums`].
	pub fn new(
		descriptor_set_allocator: Arc<dyn vk::DescriptorSetAllocator>,
		buffer_memory_allocator: Arc<vk::StandardMemoryAllocator>,
	) -> Result<Self, Error> {
		let device = descriptor_set_allocator.device();

		let work_group_size = device
			.physical_device()
			.properties()
			.max_compute_work_group_size[0];
		let subgroup_size = device
			.physical_device()
			.properties()
			.subgroup_size
			.unwrap_or(1);

		mod local_shader {
			vulkano_shaders::shader! {
				ty: "compute",
				path: "src/shaders/prefix_sums_local.comp",
				vulkan_version: "1.3",
				spirv_version: "1.3",
			}
		}
		let local_shader = local_shader::load(device.clone()).map_err(vk::Validated::unwrap)?;
		let specialized_local_shader = local_shader
			.specialize({
				use foldhash::HashMapExt as _;
				let mut constants = foldhash::HashMap::new();
				constants.insert(0, vk::SpecializationConstant::U32(work_group_size));
				constants.insert(1, vk::SpecializationConstant::U32(subgroup_size));
				constants
			})
			.unwrap();
		let local_entry_point = specialized_local_shader.entry_point("main").unwrap();

		mod global_shader {
			vulkano_shaders::shader! {
				ty: "compute",
				path: "src/shaders/prefix_sums_global.comp",
				vulkan_version: "1.3",
				spirv_version: "1.3",
			}
		}
		let global_shader = global_shader::load(device.clone()).map_err(vk::Validated::unwrap)?;
		let specialized_global_shader = global_shader
			.specialize({
				use foldhash::HashMapExt as _;
				let mut constants = foldhash::HashMap::new();
				constants.insert(0, vk::SpecializationConstant::U32(work_group_size));
				constants
			})
			.unwrap();
		let global_entry_point = specialized_global_shader.entry_point("main").unwrap();

		macro_rules! pipeline {
			($entry_point:expr) => {{
				let local_stage = vk::PipelineShaderStageCreateInfo::new($entry_point);
				let local_layout = vk::PipelineLayout::new(
					device.clone(),
					vk::PipelineDescriptorSetLayoutCreateInfo::from_stages([&local_stage])
						.into_pipeline_layout_create_info(device.clone())
						.map_err(|e| e.error.unwrap())?,
				)
				.map_err(vk::Validated::unwrap)?;

				vk::ComputePipeline::new(
					device.clone(),
					None,
					vk::ComputePipelineCreateInfo::stage_layout(local_stage, local_layout),
				)
				.map_err(vk::Validated::unwrap)?
			}};
		}
		let local_pipeline = pipeline!(local_entry_point);
		let global_pipeline = pipeline!(global_entry_point);

		Ok(Self {
			descriptor_set_allocator,
			buffer_memory_allocator,
			local_pipeline,
			global_pipeline,
			work_group_size,
		})
	}

	/// Returns the required length of these kernels’ auxiliary buffer for the
	/// given input values length, to optionally be passed to [`record`] below.
	///
	/// [`record`]: PrefixSums::record
	pub fn aux_buffer_len(&self, vals_len: u64) -> u64 {
		self.parallel_reduce_buffer_len(vals_len)
	}

	/// Records the prefix sums kernel(s) onto the given command buffer
	/// (multiple if needed to parallel-reduce over multiple work groups).
	///
	/// After execution `vals_buffer` will hold the prefix sums over its previous
	/// contents; the contents of `aux_buffer` after execution are undefined.
	///
	/// The number of elements in `vals_buffer` is derived from its length, so
	/// it must be sized exactly. `aux_buffer` is allowed to be larger than
	/// strictly necessary.
	pub fn record<L>(
		&self,
		command_buffer_builder: &mut vk::AutoCommandBufferBuilder<L>,
		vals_buffer: vk::Subbuffer<[u32]>,
		aux_buffer: Option<vk::Subbuffer<[u32]>>,
	) -> Result<(), Error> {
		assert_eq!(
			command_buffer_builder.device(),
			self.descriptor_set_allocator.device()
		);

		if vals_buffer.len() == 0 {
			return Ok(());
		}

		let aux_buffer = if let Some(aux_buffer) = aux_buffer {
			if aux_buffer.len() < self.aux_buffer_len(vals_buffer.len()) {
				return Err(Error::InvalidAuxiliaryLength);
			}
			aux_buffer
		} else {
			vk::Buffer::new_slice(
				self.buffer_memory_allocator.clone(),
				vk::BufferCreateInfo {
					usage: vk::BufferUsage::STORAGE_BUFFER,
					..Default::default()
				},
				vk::AllocationCreateInfo {
					memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
					..Default::default()
				},
				self.aux_buffer_len(vals_buffer.len()),
			)
			.unwrap()
		};

		self.record_inner(command_buffer_builder, vals_buffer, aux_buffer)
	}

	fn record_inner<L>(
		&self,
		command_buffer_builder: &mut vk::AutoCommandBufferBuilder<L>,
		vals_buffer: vk::Subbuffer<[u32]>,
		aux_buffer: vk::Subbuffer<[u32]>,
	) -> Result<(), Error> {
		let num_work_groups = self.partial_parallel_reduce_buffer_len(vals_buffer.len());

		// Local

		command_buffer_builder
			.bind_pipeline_compute(self.local_pipeline.clone())
			.unwrap();

		let local_descriptor_set = vk::DescriptorSet::new(
			self.descriptor_set_allocator.clone(),
			self.local_pipeline.layout().set_layouts()[0].clone(),
			[
				vk::WriteDescriptorSet::buffer(0, vals_buffer.clone()),
				vk::WriteDescriptorSet::buffer(1, aux_buffer.clone().slice(..num_work_groups)),
			],
			[],
		)
		.map_err(vk::Validated::unwrap)?;

		command_buffer_builder
			.bind_descriptor_sets(
				vk::PipelineBindPoint::Compute,
				self.local_pipeline.layout().clone(),
				0,
				local_descriptor_set,
			)
			.unwrap();

		unsafe { command_buffer_builder.dispatch([num_work_groups as u32, 1, 1]) }.unwrap();

		if num_work_groups == 1 {
			return Ok(());
		}

		// Parallel-reduce

		let split = self.parallel_reduce_buffer_split(num_work_groups);
		let (parallel_reduce_vals_buffer, parallel_reduce_aux_buffer) =
			aux_buffer.clone().split_at(split);
		let parallel_reduce_vals_buffer = parallel_reduce_vals_buffer.slice(..num_work_groups);

		self.record_inner(
			command_buffer_builder,
			parallel_reduce_vals_buffer.clone(),
			parallel_reduce_aux_buffer,
		)?;

		// Global

		command_buffer_builder
			.bind_pipeline_compute(self.global_pipeline.clone())
			.unwrap();

		let global_descriptor_set = vk::DescriptorSet::new(
			self.descriptor_set_allocator.clone(),
			self.global_pipeline.layout().set_layouts()[0].clone(),
			[
				vk::WriteDescriptorSet::buffer(0, vals_buffer),
				vk::WriteDescriptorSet::buffer(1, parallel_reduce_vals_buffer),
			],
			[],
		)
		.map_err(vk::Validated::unwrap)?;

		command_buffer_builder
			.bind_descriptor_sets(
				vk::PipelineBindPoint::Compute,
				self.local_pipeline.layout().clone(),
				0,
				global_descriptor_set,
			)
			.unwrap();

		unsafe { command_buffer_builder.dispatch([num_work_groups as u32, 1, 1]) }.unwrap();

		Ok(())
	}
}

impl ParallelReduceExt for PrefixSums {
	const WORK_GROUP_STRIDE: u64 = 4;

	fn work_group_size(&self) -> u64 {
		self.work_group_size as u64
	}

	fn buffer_offset_alignment(&self) -> u64 {
		self.buffer_memory_allocator
			.device()
			.physical_device()
			.properties()
			.min_storage_buffer_offset_alignment
			.as_devicesize()
	}
}

#[cfg(test)]
mod ones {
	mod vk {
		pub(super) use super::super::vk::*;
		pub(super) use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
		pub(super) use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
		pub(super) use vulkano::memory::allocator::{
			AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
		};
	}

	use super::*;
	fn ones(num: usize, check: impl FnOnce(vk::Subbuffer<[u32]>)) {
		crate::immediate_submit(
			|command_buffer_builder| {
				let device = command_buffer_builder.device();
				let memory_allocator =
					Arc::new(vk::StandardMemoryAllocator::new_default(device.clone()));

				let vals_buffer = vk::Buffer::from_iter::<u32, _>(
					memory_allocator.clone(),
					vk::BufferCreateInfo {
						usage: vk::BufferUsage::STORAGE_BUFFER,
						..Default::default()
					},
					vk::AllocationCreateInfo {
						memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE
							| vk::MemoryTypeFilter::HOST_RANDOM_ACCESS,
						..Default::default()
					},
					std::iter::repeat(1).take(num),
				)
				.unwrap();

				let prefix_sums = PrefixSums::new(
					Arc::new(vk::StandardDescriptorSetAllocator::new(
						device.clone(),
						Default::default(),
					)),
					memory_allocator,
				)
				.unwrap();

				prefix_sums
					.record(command_buffer_builder, vals_buffer.clone(), None)
					.unwrap();

				vals_buffer
			},
			check,
		);
	}

	#[test]
	fn small() {
		ones(1024, |vals_buffer| {
			let vals = vals_buffer.read().unwrap();

			assert_eq!(vals[0], 0);
			assert_eq!(vals[32], 32);
			assert_eq!(vals[69], 69);
			assert_eq!(vals[420], 420);
			assert_eq!(vals[1023], 1023);
		});
	}

	#[test]
	fn medium() {
		ones(33793, |vals_buffer| {
			let vals = vals_buffer.read().unwrap();

			assert_eq!(vals[0], 0);
			assert_eq!(vals[32], 32);
			assert_eq!(vals[69], 69);
			assert_eq!(vals[420], 420);
			assert_eq!(vals[1024], 1024);
			assert_eq!(vals[1337], 1337);
			assert_eq!(vals[6721], 6721);
			assert_eq!(vals[33792], 33792);
		});
	}

	#[test]
	fn large() {
		ones(2097153, |vals_buffer| {
			let vals = vals_buffer.read().unwrap();

			std::thread::sleep(std::time::Duration::from_millis(100));

			assert_eq!(vals[0], 0);
			assert_eq!(vals[32], 32);
			assert_eq!(vals[69], 69);
			assert_eq!(vals[420], 420);
			assert_eq!(vals[1024], 1024);
			assert_eq!(vals[1337], 1337);
			assert_eq!(vals[42069], 42069);
			assert_eq!(vals[69420], 69420);
			assert_eq!(vals[269800], 269800);
			assert_eq!(vals[1048576], 1048576);
			assert_eq!(vals[1482911], 1482911);
			assert_eq!(vals[2097152], 2097152);
		});
	}
}
