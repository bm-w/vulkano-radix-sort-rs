pub mod bitwise_and_or;
pub mod prefix_sums;
mod util;

#[cfg(test)]
mod vk {
	pub(super) use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
}

#[cfg(test)]
fn immediate_submit<R>(
	record: impl FnOnce(&mut vk::AutoCommandBufferBuilder<vk::PrimaryAutoCommandBuffer>) -> R,
	check: impl FnOnce(R),
) {
	use std::sync::Arc;

	mod vk {
		pub(super) use vulkano::command_buffer::{
			AutoCommandBufferBuilder, CommandBufferUsage,
			allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
		};
		pub(super) use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
		pub(super) use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
		pub(super) use vulkano::library::VulkanLibrary as Library;
	}
	use vulkano::{command_buffer::PrimaryCommandBufferAbstract as _, sync::GpuFuture as _};

	let library = vk::Library::new().unwrap();
	let instance = vk::Instance::new(
		library,
		vk::InstanceCreateInfo {
			flags: vk::InstanceCreateFlags::ENUMERATE_PORTABILITY,
			..Default::default()
		},
	)
	.unwrap();
	let physical_device = instance
		.enumerate_physical_devices()
		.unwrap()
		.next()
		.unwrap();
	let queue_family_index = physical_device
		.queue_family_properties()
		.iter()
		.position(|queue_family_properties| {
			queue_family_properties
				.queue_flags
				.contains(vk::QueueFlags::GRAPHICS)
		})
		.unwrap() as u32;
	let (device, mut queues) = vk::Device::new(
		physical_device.clone(),
		vk::DeviceCreateInfo {
			queue_create_infos: vec![vk::QueueCreateInfo {
				queue_family_index,
				..Default::default()
			}],
			..Default::default()
		},
	)
	.unwrap();
	let queue = queues.next().unwrap();

	let command_buffer_allocator = Arc::new(vk::StandardCommandBufferAllocator::new(
		device.clone(),
		vk::StandardCommandBufferAllocatorCreateInfo {
			secondary_buffer_count: 1,
			..Default::default()
		},
	));

	let mut command_buffer_builder = vk::AutoCommandBufferBuilder::primary(
		command_buffer_allocator.clone(),
		queue_family_index,
		vk::CommandBufferUsage::OneTimeSubmit,
	)
	.unwrap();

	let result = record(&mut command_buffer_builder);

	command_buffer_builder
		.build()
		.unwrap()
		.execute(queue)
		.unwrap()
		.then_signal_fence_and_flush()
		.unwrap()
		.wait(None)
		.unwrap();

	check(result);
}
