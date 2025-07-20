/// Helper methods for parallel reduction over multiple work groups.
pub(crate) trait ParallelReduceExt {
	const WORK_GROUP_STRIDE: u64;

	/// Returns the size of a work group.
	fn work_group_size(&self) -> u64;

	/// Returns the alignment for buffer offsets.
	fn buffer_offset_alignment(&self) -> u64;

	/// Returns the number of work groups needed to process the given number
	/// of values.
	fn partial_parallel_reduce_buffer_len(&self, vals_len: u64) -> u64 {
		(vals_len - 1) / self.work_group_size() + 1
	}

	/// Returns the split offset for the next parallel-reduction buffer for the
	/// given number of work groups.
	fn parallel_reduce_buffer_split(&self, num_work_groups: u64) -> u64 {
		let align = self.buffer_offset_alignment();
		parallel_reduce_buffer_split(num_work_groups, Self::WORK_GROUP_STRIDE, align)
	}

	/// Returns the total length of the multi-level parallel-reduction buffer
	/// for the given number of values.
	fn parallel_reduce_buffer_len(&self, mut vals_len: u64) -> u64 {
		let align = self.buffer_offset_alignment();
		let mut next_buffer_split = 0;
		let mut min_buffer_len = 0;
		while vals_len > 1 {
			min_buffer_len += next_buffer_split;
			let num_work_groups = self.partial_parallel_reduce_buffer_len(vals_len);
			next_buffer_split =
				parallel_reduce_buffer_split(num_work_groups, Self::WORK_GROUP_STRIDE, align);
			vals_len = num_work_groups;
		}
		min_buffer_len + 1
	}

	/// Returns the number of parallel-reduction levels for the given number
	/// of values.
	fn parallel_reduce_depth(&self, mut vals_len: u64) -> u64 {
		let mut depth = 0;
		while vals_len > 1 {
			depth += 1;
			vals_len = self.partial_parallel_reduce_buffer_len(vals_len);
		}
		depth
	}
}

fn parallel_reduce_buffer_split(len: u64, stride: u64, align: u64) -> u64 {
	((len * stride - 1) / align + 1) * align / stride
}
