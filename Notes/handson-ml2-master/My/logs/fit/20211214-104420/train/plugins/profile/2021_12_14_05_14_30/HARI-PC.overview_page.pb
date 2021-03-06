?	????kW??????kW??!????kW??	???@Y@???@Y@!???@Y@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:????kW??Q?l???A??|?????YB???8??rEagerKernelExecute 0*	??S㥣i@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate_&??????!k? ?K?D@)????c???1??n/?B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatW`??Vϱ?!!?:?w?@@)<?$???19??z??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?GS=???!??=b_lU@)?xy:W???1+sX?!@:Preprocessing2U
Iterator::Model::ParallelMapV2?/?????!-?c/!@)?/?????1-?c/!@:Preprocessing2F
Iterator::Model?X4????!????,@)zR&5???1i?B?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?Eж???!?Ր?l?@)?Eж???1?Ր?l?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?CR%?s?!V0??@)?CR%?s?1V0??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????Ŋ??!&$5wE@)+N?f?m?1e?j ?6??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 35.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???@Y@I!???5X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Q?l???Q?l???!Q?l???      ??!       "      ??!       *      ??!       2	??|???????|?????!??|?????:      ??!       B      ??!       J	B???8??B???8??!B???8??R      ??!       Z	B???8??B???8??!B???8??b      ??!       JCPU_ONLYY???@Y@b q!???5X@Y      Y@q???_??P@"?	
both?Your program is POTENTIALLY input-bound because 35.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?66.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 