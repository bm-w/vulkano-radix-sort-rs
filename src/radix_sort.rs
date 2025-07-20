use std::{ops::Range, sync::Arc};

mod vk {
	pub(super) use vulkano::buffer::{
		Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
	};
	pub(super) use vulkano::command_buffer::{AutoCommandBufferBuilder, DispatchIndirectCommand};
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

use crate::{
	bitwise_and_or::{BitwiseAndOr, Error as BitwiseAndOrError},
	prefix_sums::{self, Error as PrefixSumsError, PrefixSums},
	util::ParallelReduceExt,
};

/// A set of compute kernels that sorts a buffer of 32-bit unsigned integers,
/// or arbitrary values with 32-bit unsigned integer keys, in-place.
///
/// `R_WAY` is the number of ways each value of key is sorted during each radix
/// round. A higher value will result in fewer rounds, but more work per round
/// and more auxiliary memory usage. `R_WAY` must be a power of two.
pub struct RadixSort<const R_WAY: u32 = 4> {
	bitwise_and_or: BitwiseAndOr,
	prefix_sums: PrefixSums,
	indirect_pipeline: Arc<vk::ComputePipeline>,
	local_pipeline: Arc<vk::ComputePipeline>,
	global_pipeline: Arc<vk::ComputePipeline>,
	copy_pipeline: Arc<vk::ComputePipeline>,
	work_group_size: u64,
}

/// The error type that can be returned by [`RadixSort`]’s methods
#[derive(thiserror::Error, Debug)]
pub enum Error {
	#[error("The values buffer has invalid length")]
	InvalidValuesLength,
	#[error("The auxiliary buffer has invalid length")]
	InvalidAuxiliaryLength,
	#[error(transparent)]
	Vulkan(#[from] vk::Error),
}

impl<const R_WAY: u32> RadixSort<R_WAY> {
	/// Constructs a new instance of [`RadixSort`].
	pub fn new(
		descriptor_set_allocator: Arc<dyn vk::DescriptorSetAllocator>,
		buffer_memory_allocator: Arc<vk::StandardMemoryAllocator>,
	) -> Result<Self, Error> {
		let bitwise_and_or = BitwiseAndOr::new(descriptor_set_allocator.clone())
			.map_err(bitwise_and_or_vulkan_error)?;
		let prefix_sums = PrefixSums::new(descriptor_set_allocator, buffer_memory_allocator)
			.map_err(prefix_sums_vulkan_error)?;

		let device = prefix_sums.buffer_memory_allocator.device();

		let work_group_size = prefix_sums.work_group_size;
		let subgroup_size = device
			.physical_device()
			.properties()
			.subgroup_size
			.unwrap_or(1);
		assert!(subgroup_size >= 32); // TODO: Relax?
		let specialization_constants = {
			use foldhash::HashMapExt as _;
			let mut constants = foldhash::HashMap::new();
			constants.insert(0, vk::SpecializationConstant::U32(work_group_size));
			constants.insert(1, vk::SpecializationConstant::U32(subgroup_size));
			constants.insert(2, vk::SpecializationConstant::U32(R_WAY));
			constants
		};

		macro_rules! create_pipeline {
			($path:literal) => {{
				mod shader {
					vulkano_shaders::shader! {
						ty: "compute",
						path: $path,
						vulkan_version: "1.3",
						spirv_version: "1.3",
					}
				}
				let shader = shader::load(device.clone()).map_err(vk::Validated::unwrap)?;
				let specialized_shader =
					shader.specialize(specialization_constants.clone()).unwrap();
				let entry_point = specialized_shader.entry_point("main").unwrap();

				let stage = vk::PipelineShaderStageCreateInfo::new(entry_point);
				let layout = vk::PipelineLayout::new(
					device.clone(),
					vk::PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
						.into_pipeline_layout_create_info(device.clone())
						.map_err(|e| e.error.unwrap())?,
				)
				.map_err(vk::Validated::unwrap)?;

				vk::ComputePipeline::new(
					device.clone(),
					None,
					vk::ComputePipelineCreateInfo::stage_layout(stage, layout),
				)
				.map_err(vk::Validated::unwrap)?
			}};
		}
		let indirect_pipeline = create_pipeline!("src/shaders/radix_sort_indirect.comp");
		let local_pipeline = create_pipeline!("src/shaders/radix_sort_local.comp");
		let global_pipeline = create_pipeline!("src/shaders/radix_sort_global.comp");
		let copy_pipeline = create_pipeline!("src/shaders/radix_sort_copy.comp");

		Ok(Self {
			bitwise_and_or,
			prefix_sums,
			indirect_pipeline,
			local_pipeline,
			global_pipeline,
			copy_pipeline,
			work_group_size: work_group_size as u64,
		})
	}

	fn radix_rounds(&self) -> u64 {
		(8 * size_of::<u32>() as u64) / R_WAY.trailing_zeros() as u64
	}

	fn aux_buffer_ranges(&self, vals_len: u64) -> AuxBufferRanges {
		let align = self.bitwise_and_or.buffer_offset_alignment();
		let u32_stride = size_of::<u32>() as u64;

		let bitwise_and_or = self.bitwise_and_or.results_buffer_len(vals_len);
		let aligned_bitwise_and_or =
			(((bitwise_and_or * u32_stride - 1) / align + 1) * align) / u32_stride;

		let prefix_sums_vals = R_WAY as u64 * ((vals_len - 1) / self.work_group_size + 1);
		let aligned_prefix_sums_vals =
			(((prefix_sums_vals * u32_stride - 1) / align + 1) * align) / u32_stride;
		let prefix_sums_aux = self
			.prefix_sums
			.parallel_reduce_buffer_len(prefix_sums_vals);
		let aligned_prefix_sums_aux =
			(((prefix_sums_aux * u32_stride - 1) / align + 1) * align) / u32_stride;
		let prefix_sums_depth = self.prefix_sums.parallel_reduce_depth(prefix_sums_vals);

		let indirect_dispatches_offset =
			(2 * aligned_bitwise_and_or).max(aligned_prefix_sums_vals + aligned_prefix_sums_aux);
		let aligned_indirect_dispatches_offset =
			(((indirect_dispatches_offset * u32_stride - 1) / align + 1) * align) / u32_stride;
		let num_indirect_dispatches_per_round = 1 + prefix_sums_depth;
		let num_indirect_dispatches =
			2 * num_indirect_dispatches_per_round * self.radix_rounds() + 1;
		assert!(align >= size_of::<vk::DispatchIndirectCommand>() as u64);
		let indirect_dispatches = aligned_indirect_dispatches_offset
			..aligned_indirect_dispatches_offset + (num_indirect_dispatches * align) / u32_stride;

		AuxBufferRanges {
			bitwise_and: 0..bitwise_and_or,
			bitwise_or: aligned_bitwise_and_or..aligned_bitwise_and_or + bitwise_and_or,
			prefix_sums_vals: 0..prefix_sums_vals,
			prefix_sums_aux: aligned_prefix_sums_vals..aligned_prefix_sums_vals + prefix_sums_aux,
			indirect_dispatches: indirect_dispatches.clone(),
			total_len: indirect_dispatches.end,
			depth: prefix_sums_depth,
			align,
		}
	}

	/// Returns the required length of the kernels’ auxiliary buffer for the
	/// given input values length, to optionally be passed to [`record`] below.
	///
	/// [`record`]: RadixSort::record
	pub fn aux_buffer_len(&self, vals_len: u64) -> u64 {
		self.aux_buffer_ranges(vals_len).total_len
	}

	// TODO: Make  more ergonomic for keys-only sort, i.e. not requiring specified `T`
	/// Records the radix sort kernels onto the given command buffer (in
	/// multiple rounds depending on `R_WAY`).
	///
	/// After execution `keys_buffer` and `vals_buffer` will hold the sorted
	/// values and keys.
	///
	/// The number of elements in `keys_buffer` is derived from its length, so
	/// it must be sized exactly, and must match between keys and values.
	/// `aux_buffer` may be larger than strictly necessary.
	pub fn record<L, T>(
		&self,
		command_buffer_builder: &mut vk::AutoCommandBufferBuilder<L>,
		keys_buffer: vk::Subbuffer<[u32]>,
		vals_buffer: Option<vk::Subbuffer<[T]>>,
		aux_buffer: Option<vk::Subbuffer<[u32]>>,
	) -> Result<(), Error>
	where
		T: vk::BufferContents,
	{
		if keys_buffer.len() == 0 {
			return Ok(());
		}

		if vals_buffer
			.as_ref()
			.is_some_and(|b| b.len() != keys_buffer.len())
		{
			return Err(Error::InvalidValuesLength);
		}

		let val_size = vals_buffer.as_ref().map(|_| size_of::<T>()).unwrap_or(0) as u32;

		let aux_buffer_ranges = self.aux_buffer_ranges(keys_buffer.len() as u64);

		let aux_buffer = if let Some(aux_buffer) = aux_buffer {
			if aux_buffer.len() < aux_buffer_ranges.total_len {
				return Err(Error::InvalidAuxiliaryLength);
			}
			aux_buffer
		} else {
			vk::Buffer::new_slice(
				self.prefix_sums.buffer_memory_allocator.clone(),
				vk::BufferCreateInfo {
					usage: vk::BufferUsage::STORAGE_BUFFER
						| vk::BufferUsage::INDIRECT_BUFFER
						| vk::BufferUsage::TRANSFER_SRC,
					..Default::default()
				},
				vk::AllocationCreateInfo {
					memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
					..Default::default()
				},
				aux_buffer_ranges.total_len,
			)
			.unwrap() // TODO: Better error handling?
		};

		// TODO: (`Option`) parameters?
		let doubled_keys_buffer = vk::Buffer::new_slice::<u32>(
			self.prefix_sums.buffer_memory_allocator.clone(),
			vk::BufferCreateInfo {
				usage: vk::BufferUsage::STORAGE_BUFFER | vk::BufferUsage::TRANSFER_SRC,
				..Default::default()
			},
			vk::AllocationCreateInfo {
				memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
				..Default::default()
			},
			keys_buffer.len(),
		)
		.unwrap(); // TODO: Better error handling?

		let vals_buffer: Option<vk::Subbuffer<[u8]>> = vals_buffer.map(|b| b.reinterpret());
		// TODO: (`Option`) parameters?
		let doubled_vals_buffer = vals_buffer.as_ref().map(|b| {
			vk::Buffer::new_slice::<u8>(
				self.prefix_sums.buffer_memory_allocator.clone(),
				vk::BufferCreateInfo {
					usage: vk::BufferUsage::STORAGE_BUFFER | vk::BufferUsage::TRANSFER_SRC,
					..Default::default()
				},
				vk::AllocationCreateInfo {
					memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE,
					..Default::default()
				},
				b.len(),
			)
			.unwrap()
		}); // TODO: Better error handling?

		let bitwise_and_or_result_offset = self
			.bitwise_and_or
			.record(
				command_buffer_builder,
				keys_buffer.clone(),
				aux_buffer
					.clone()
					.slice(aux_buffer_ranges.bitwise_and.clone()),
				aux_buffer
					.clone()
					.slice(aux_buffer_ranges.bitwise_or.clone()),
			)
			.map_err(bitwise_and_or_vulkan_error)?;

		command_buffer_builder
			.bind_pipeline_compute(self.indirect_pipeline.clone())
			.unwrap()
			.bind_descriptor_sets(
				vk::PipelineBindPoint::Compute,
				self.indirect_pipeline.layout().clone(),
				0,
				vk::DescriptorSet::new(
					self.prefix_sums.descriptor_set_allocator.clone(),
					self.indirect_pipeline.layout().set_layouts()[0].clone(),
					[
						vk::WriteDescriptorSet::buffer(
							0,
							aux_buffer.clone().slice({
								let o = aux_buffer_ranges.bitwise_and.start
									+ bitwise_and_or_result_offset;
								o..o + 1
							}),
						),
						vk::WriteDescriptorSet::buffer(
							1,
							aux_buffer.clone().slice({
								let o = aux_buffer_ranges.bitwise_or.start
									+ bitwise_and_or_result_offset;
								o..o + 1
							}),
						),
						vk::WriteDescriptorSet::buffer(
							2,
							aux_buffer
								.clone()
								.slice(aux_buffer_ranges.indirect_dispatches.clone()),
						),
					],
					[],
				)
				.map_err(vk::Validated::unwrap)?,
			)
			.unwrap()
			.push_constants(
				self.indirect_pipeline.layout().clone(),
				0,
				keys_buffer.len() as u32,
			)
			.unwrap();
		unsafe { command_buffer_builder.dispatch([1, 1, 1]) }.unwrap();

		let local_descriptor_sets = transpose_arr(std::array::from_fn::<_, 2, _>(|i| {
			let keys = if i == 0 {
				keys_buffer.clone()
			} else {
				doubled_keys_buffer.clone()
			};

			vk::DescriptorSet::new(
				self.prefix_sums.descriptor_set_allocator.clone(),
				self.local_pipeline.layout().set_layouts()[0].clone(),
				[
					vk::WriteDescriptorSet::buffer(0, keys.clone()),
					vk::WriteDescriptorSet::buffer(
						1,
						aux_buffer
							.clone()
							.slice(aux_buffer_ranges.prefix_sums_vals.clone()),
					),
				],
				[],
			)
			.map_err(vk::Validated::unwrap)
		}))?;

		let prefix_sums_dispatches = {
			let mut dispatches = Vec::new();
			self.prefix_sums
				.create_direct_dispatches(
					&mut dispatches,
					aux_buffer
						.clone()
						.slice(aux_buffer_ranges.prefix_sums_vals.clone()),
					aux_buffer
						.clone()
						.slice(aux_buffer_ranges.prefix_sums_aux.clone()),
				)
				.map_err(prefix_sums_vulkan_error)?;
			[dispatches.clone(), dispatches]
		};

		let global_descriptor_sets = transpose_arr(std::array::from_fn::<_, 2, _>(|i| {
			let (keys_in, keys_out, vals_in, vals_out) = if i == 0 {
				(
					keys_buffer.clone(),
					doubled_keys_buffer.clone(),
					vals_buffer.clone(),
					doubled_vals_buffer.clone(),
				)
			} else {
				(
					doubled_keys_buffer.clone(),
					keys_buffer.clone(),
					doubled_vals_buffer.clone(),
					vals_buffer.clone(),
				)
			};

			vk::DescriptorSet::new(
				self.prefix_sums.descriptor_set_allocator.clone(),
				self.global_pipeline.layout().set_layouts()[0].clone(),
				[
					vk::WriteDescriptorSet::buffer(0, keys_in.clone()),
					vk::WriteDescriptorSet::buffer(
						1,
						vals_in.unwrap_or_else(|| keys_in.reinterpret()),
					),
					vk::WriteDescriptorSet::buffer(2, keys_out.clone()),
					vk::WriteDescriptorSet::buffer(
						3,
						vals_out.unwrap_or_else(|| keys_out.reinterpret()),
					),
					vk::WriteDescriptorSet::buffer(
						4,
						aux_buffer
							.clone()
							.slice(aux_buffer_ranges.prefix_sums_vals.clone()),
					),
				],
				[],
			)
			.map_err(vk::Validated::unwrap)
		}))?;

		let stride = aux_buffer_ranges.align / size_of::<u32>() as u64;
		for i in 0..((8 * size_of::<u32>() - 1) / (R_WAY - 1).count_ones() as usize + 1) {
			for j in 0..=1 {
				let offset_r = aux_buffer_ranges.indirect_dispatches.start
					+ (2 * i + j) as u64 * stride * (1 + aux_buffer_ranges.depth);

				// Local
				command_buffer_builder
					.bind_pipeline_compute(self.local_pipeline.clone())
					.unwrap()
					.bind_descriptor_sets(
						vk::PipelineBindPoint::Compute,
						self.local_pipeline.layout().clone(),
						0,
						local_descriptor_sets[i % 2].clone(),
					)
					.unwrap()
					.push_constants(self.local_pipeline.layout().clone(), 0, i as u32)
					.unwrap();
				unsafe {
					command_buffer_builder.dispatch_indirect(
						aux_buffer
							.clone()
							.slice(offset_r..offset_r + 3)
							.reinterpret(),
					)
				}
				.unwrap();

				// Prefix Sums
				self.prefix_sums.record_dispatches(
					command_buffer_builder,
					prefix_sums_dispatches[i % 2]
						.iter()
						.enumerate()
						.map(|(i, (_, ds))| {
							let dist = (aux_buffer_ranges.depth - 1).abs_diff(i as u64);
							let offset = offset_r + stride * (aux_buffer_ranges.depth - dist);
							(
								prefix_sums::Dispatch::Indirect(
									aux_buffer.clone().slice(offset..offset + 3).reinterpret(),
								),
								ds.clone(),
							)
						}),
				);

				// Global
				command_buffer_builder
					.bind_pipeline_compute(self.global_pipeline.clone())
					.unwrap()
					.bind_descriptor_sets(
						vk::PipelineBindPoint::Compute,
						self.global_pipeline.layout().clone(),
						0,
						global_descriptor_sets[i % 2].clone(),
					)
					.unwrap()
					.push_constants(
						self.global_pipeline.layout().clone(),
						0,
						[i as u32, val_size],
					)
					.unwrap();
				unsafe {
					command_buffer_builder.dispatch_indirect(
						aux_buffer
							.clone()
							.slice(offset_r..offset_r + 3)
							.reinterpret(),
					)
				}
				.unwrap();
			}
		}

		command_buffer_builder
			.bind_pipeline_compute(self.copy_pipeline.clone())
			.unwrap()
			.bind_descriptor_sets(
				vk::PipelineBindPoint::Compute,
				self.copy_pipeline.layout().clone(),
				0,
				vk::DescriptorSet::new(
					self.prefix_sums.descriptor_set_allocator.clone(),
					self.copy_pipeline.layout().set_layouts()[0].clone(),
					[
						vk::WriteDescriptorSet::buffer(0, doubled_keys_buffer.clone()),
						vk::WriteDescriptorSet::buffer(
							1,
							doubled_vals_buffer
								.unwrap_or_else(|| doubled_keys_buffer.reinterpret()),
						),
						vk::WriteDescriptorSet::buffer(2, keys_buffer.clone()),
						vk::WriteDescriptorSet::buffer(
							3,
							vals_buffer.unwrap_or_else(|| keys_buffer.reinterpret()),
						),
					],
					[],
				)
				.map_err(vk::Validated::unwrap)?,
			)
			.unwrap()
			.push_constants(self.indirect_pipeline.layout().clone(), 0, val_size)
			.unwrap();
		unsafe {
			command_buffer_builder.dispatch_indirect({
				let o = aux_buffer_ranges.indirect_dispatches.end - 4;
				aux_buffer.slice(o..o + 3).reinterpret()
			})
		}
		.unwrap();

		Ok(())
	}
}

#[derive(Debug, Default)]
struct AuxBufferRanges {
	bitwise_and: Range<u64>,
	bitwise_or: Range<u64>,
	prefix_sums_vals: Range<u64>,
	prefix_sums_aux: Range<u64>,
	/// In terms of `align_of::<vk::DispatchIndirectCommand>()`s (`u32`s).
	indirect_dispatches: Range<u64>,
	/// In terms of `u32`s.
	total_len: u64,
	depth: u64,
	/// In bytes.
	align: u64,
}

fn bitwise_and_or_vulkan_error(e: BitwiseAndOrError) -> Error {
	let BitwiseAndOrError::Vulkan(e) = e else {
		unreachable!()
	};
	e.into()
}

fn prefix_sums_vulkan_error(e: PrefixSumsError) -> Error {
	let PrefixSumsError::Vulkan(e) = e else {
		unreachable!()
	};
	e.into()
}

fn transpose_arr<T, E, const N: usize>(arr: [Result<T, E>; N]) -> Result<[T; N], E> {
	let mut out = std::array::from_fn::<_, N, _>(|_| None);
	for (i, res) in arr.into_iter().enumerate() {
		out[i] = Some(res?);
	}
	Ok(out.map(Option::unwrap))
}

#[cfg(test)]
mod shuffled {
	mod vk {
		pub(super) use super::super::vk::*;
		pub(super) use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
		pub(super) use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
		pub(super) use vulkano::memory::allocator::{
			AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
		};
	}

	use super::*;

	fn shuffled(num: usize, stride: usize, check: impl FnOnce(vk::Subbuffer<[u32]>)) {
		crate::immediate_submit(
			|command_buffer_builder| {
				let device = command_buffer_builder.device();
				let memory_allocator =
					Arc::new(vk::StandardMemoryAllocator::new_default(device.clone()));

				let radix_sort = RadixSort::<4>::new(
					Arc::new(vk::StandardDescriptorSetAllocator::new(
						device.clone(),
						Default::default(),
					)),
					memory_allocator.clone(),
				)
				.unwrap();

				let keys_buffer = vk::Buffer::from_iter::<u32, _>(
					memory_allocator.clone(),
					vk::BufferCreateInfo {
						usage: vk::BufferUsage::STORAGE_BUFFER | vk::BufferUsage::TRANSFER_DST,
						..Default::default()
					},
					vk::AllocationCreateInfo {
						memory_type_filter: vk::MemoryTypeFilter::PREFER_DEVICE
							| vk::MemoryTypeFilter::HOST_RANDOM_ACCESS,
						..Default::default()
					},
					(0..num).map(|i| (((i + 1) * stride - 1) % num) as u32),
				)
				.unwrap();

				radix_sort
					.record::<_, ()>(command_buffer_builder, keys_buffer.clone(), None, None)
					.unwrap();

				keys_buffer
			},
			check,
		);
	}

	#[test]
	fn tiny() {
		shuffled(137, 91, |keys| {
			let keys = keys.read().unwrap();
			for (i, key) in keys.iter().enumerate() {
				assert_eq!(*key as usize, i);
			}
		});
	}

	#[test]
	fn small() {
		shuffled(1337, 891, |keys| {
			let keys = keys.read().unwrap();
			for (i, key) in keys.iter().enumerate() {
				assert_eq!(*key as usize, i);
			}
		});
	}

	#[test]
	fn medium() {
		shuffled(69421, 46281, |keys| {
			let keys = keys.read().unwrap();
			for (i, key) in keys.iter().enumerate() {
				assert_eq!(*key as usize, i);
			}
		});
	}

	#[test]
	fn large() {
		shuffled(1337691, 891793, |keys| {
			let keys = keys.read().unwrap();
			for (i, key) in keys.iter().enumerate() {
				assert_eq!(*key as usize, i);
			}
		});
	}
}
