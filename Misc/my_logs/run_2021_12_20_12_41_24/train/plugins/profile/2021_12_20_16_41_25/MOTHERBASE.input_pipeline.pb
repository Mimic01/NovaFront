	X?5?;N??X?5?;N??!X?5?;N??	\0x?
@\0x?
@!\0x?
@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X?5?;N??~??k	???AO??e???Y?{??Pk??*	gffff&M@2F
Iterator::Model???????!?D|s?TM@))\???(??1萚`??G@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF??_??!?՟?bi4@)?j+??݃?1CX?#Y?0@:Preprocessing2U
Iterator::Model::ParallelMapV2S?!?uq{?!bΆK?&@)S?!?uq{?1bΆK?&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateŏ1w-!?!9?5Jq*@){?G?zt?1?#Y?'!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip=?U?????!????@?D@)F%u?k?1??V?9?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?Le?!PݸM??@)??_?Le?1PݸM??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!??sHM0@)/n??b?1??sHM0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??y?):??!?ɣ??.@)??_?LU?1PݸM??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s6.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9\0x?
@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~??k	???~??k	???!~??k	???      ??!       "      ??!       *      ??!       2	O??e???O??e???!O??e???:      ??!       B      ??!       J	?{??Pk???{??Pk??!?{??Pk??R      ??!       Z	?{??Pk???{??Pk??!?{??Pk??JCPU_ONLYY\0x?
@b 