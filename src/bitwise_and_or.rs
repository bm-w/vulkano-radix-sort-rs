use std::sync::Arc;

mod vk {
	pub(super) use vulkano::buffer::Subbuffer;
	pub(super) use vulkano::command_buffer::AutoCommandBufferBuilder;
	pub(super) use vulkano::descriptor_set::{
		DescriptorSet, WriteDescriptorSet, allocator::DescriptorSetAllocator,
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

/// A compute kernel that performs bitwise AND/OR operations on an array of
/// 32-bit unsigned integers.
pub struct BitwiseAndOr {
	descriptor_set_allocator: Arc<dyn vk::DescriptorSetAllocator>,
	pipeline: Arc<vk::ComputePipeline>,
	work_group_size: u32,
}

/// The error type that can be returned by [`BitwiseAndOr`]’s methods.
#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("The values buffer is empty")]
	ValsEmpty,
	#[error("The results buffer has invalid length")]
	InvalidResultsLength,
	#[error(transparent)]
	Vulkan(#[from] vk::Error),
}

impl BitwiseAndOr {
	/// Constructs a new instance of [`BitwiseAndOr`].
	pub fn new(
		descriptor_set_allocator: Arc<dyn vk::DescriptorSetAllocator>,
	) -> Result<Self, Error> {
		let device = descriptor_set_allocator.device();

		mod shader {
			vulkano_shaders::shader! {
				ty: "compute",
				path: "src/shaders/bitwise_and_or.comp",
				vulkan_version: "1.3",
				spirv_version: "1.3",
			}
		}

		let work_group_size = device
			.physical_device()
			.properties()
			.max_compute_work_group_size[0];
		let subgroup_size = device
			.physical_device()
			.properties()
			.subgroup_size
			.unwrap_or(1);

		let shader = shader::load(device.clone()).map_err(vk::Validated::unwrap)?;
		let specialized_shader = shader
			.specialize({
				use foldhash::HashMapExt as _;
				let mut constants = foldhash::HashMap::new();
				constants.insert(0, vk::SpecializationConstant::U32(work_group_size));
				constants.insert(1, vk::SpecializationConstant::U32(subgroup_size));
				constants
			})
			.unwrap();
		let entry_point = specialized_shader.entry_point("main").unwrap();

		let stage = vk::PipelineShaderStageCreateInfo::new(entry_point);
		let layout = vk::PipelineLayout::new(
			device.clone(),
			vk::PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
				.into_pipeline_layout_create_info(device.clone())
				.map_err(|e| e.error.unwrap())?,
		)
		.map_err(vk::Validated::unwrap)?;

		let pipeline = vk::ComputePipeline::new(
			device.clone(),
			None,
			vk::ComputePipelineCreateInfo::stage_layout(stage, layout),
		)
		.map_err(vk::Validated::unwrap)?;

		Ok(Self {
			descriptor_set_allocator,
			pipeline,
			work_group_size,
		})
	}

	/// Returns the required length of each of this kernel’s results buffers
	/// for the given input values length.
	pub fn results_buffer_len(&self, vals_len: u64) -> u64 {
		self.parallel_reduce_buffer_len(vals_len)
	}

	/// Records the bitwise-AND/OR kernel(s) onto the given command buffer
	/// (multiple if needed to parallel-reduce over multiple work groups).
	///
	/// Returns the offset in `and_results_buffer` and `or_results_buffer`
	/// where the final results can be found after execution.
	pub fn record<L>(
		&self,
		command_buffer_builder: &mut vk::AutoCommandBufferBuilder<L>,
		vals_buffer: vk::Subbuffer<[u32]>,
		and_results_buffer: vk::Subbuffer<[u32]>,
		or_results_buffer: vk::Subbuffer<[u32]>,
	) -> Result<u64, Error> {
		assert_eq!(
			command_buffer_builder.device(),
			self.descriptor_set_allocator.device()
		);

		if vals_buffer.len() == 0 {
			return Err(Error::ValsEmpty);
		}

		let min_result_buffers_len = self.results_buffer_len(vals_buffer.len());
		if and_results_buffer.len() < min_result_buffers_len
			|| or_results_buffer.len() < min_result_buffers_len
		{
			return Err(Error::InvalidResultsLength);
		}

		command_buffer_builder
			.bind_pipeline_compute(self.pipeline.clone())
			.unwrap();

		self.record_inner(
			command_buffer_builder,
			vals_buffer.clone(),
			vals_buffer,
			and_results_buffer,
			or_results_buffer,
			self.work_group_size,
		)
	}

	fn record_inner<L>(
		&self,
		command_buffer_builder: &mut vk::AutoCommandBufferBuilder<L>,
		and_vals_buffer: vk::Subbuffer<[u32]>,
		or_vals_buffer: vk::Subbuffer<[u32]>,
		and_results_buffer: vk::Subbuffer<[u32]>,
		or_results_buffer: vk::Subbuffer<[u32]>,
		work_group_size: u32,
	) -> Result<u64, Error> {
		let descriptor_set = vk::DescriptorSet::new(
			self.descriptor_set_allocator.clone(),
			self.pipeline.layout().set_layouts()[0].clone(),
			[
				vk::WriteDescriptorSet::buffer(0, and_vals_buffer.clone()),
				vk::WriteDescriptorSet::buffer(1, or_vals_buffer.clone()),
				vk::WriteDescriptorSet::buffer(2, and_results_buffer.clone()),
				vk::WriteDescriptorSet::buffer(3, or_results_buffer.clone()),
			],
			[],
		)
		.map_err(vk::Validated::unwrap)?;

		command_buffer_builder
			.bind_descriptor_sets(
				vk::PipelineBindPoint::Compute,
				self.pipeline.layout().clone(),
				0,
				descriptor_set,
			)
			.unwrap();

		let num_work_groups = self.partial_parallel_reduce_buffer_len(and_vals_buffer.len());
		unsafe { command_buffer_builder.dispatch([num_work_groups as u32, 1, 1]) }.unwrap();

		if num_work_groups == 1 {
			return Ok(0);
		}

		let split = self.parallel_reduce_buffer_split(num_work_groups);
		let (and_vals_buffer, and_results_buffer) = and_results_buffer.split_at(split);
		let (or_vals_buffer, or_results_buffer) = or_results_buffer.split_at(split);

		Ok(split
			+ self.record_inner(
				command_buffer_builder,
				and_vals_buffer.slice(..num_work_groups),
				or_vals_buffer.slice(..num_work_groups),
				and_results_buffer,
				or_results_buffer,
				work_group_size,
			)?)
	}
}

impl ParallelReduceExt for BitwiseAndOr {
	const WORK_GROUP_STRIDE: u64 = 4;

	fn work_group_size(&self) -> u64 {
		self.work_group_size as u64
	}

	fn buffer_offset_alignment(&self) -> u64 {
		self.pipeline
			.device()
			.physical_device()
			.properties()
			.min_storage_buffer_offset_alignment
			.as_devicesize()
	}
}

#[test]
fn big() {
	mod vk {
		pub(super) use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
		pub(super) use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
		pub(super) use vulkano::memory::allocator::{
			AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
		};
	}

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
						| vk::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
					..Default::default()
				},
				0..1048577,
			)
			.unwrap();

			let [and_results_buffer, or_results_buffer] = std::array::from_fn(|_| {
				vk::Buffer::new_slice::<u32>(
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
					1033,
				)
				.unwrap()
			});

			let bitwise_and_or = BitwiseAndOr::new(Arc::new(
				vk::StandardDescriptorSetAllocator::new(device.clone(), Default::default()),
			))
			.unwrap();

			let results_offset = bitwise_and_or
				.record(
					command_buffer_builder,
					vals_buffer,
					and_results_buffer.clone(),
					or_results_buffer.clone(),
				)
				.unwrap();

			(and_results_buffer, or_results_buffer, results_offset)
		},
		|(and_results_buffer, or_results_buffer, results_offset)| {
			let and_results = and_results_buffer.read().unwrap();
			let or_results = or_results_buffer.read().unwrap();

			assert_eq!(results_offset, 1032);
			assert_eq!(and_results[1032], 0);
			assert_eq!(or_results[1032], 2097151);
		},
	);
}
