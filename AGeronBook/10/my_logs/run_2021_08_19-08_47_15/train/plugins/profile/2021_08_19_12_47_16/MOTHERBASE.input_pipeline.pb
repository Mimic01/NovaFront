	????????????!??????	?ma0l[0@?ma0l[0@!?ma0l[0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???????f??j+??A??{??P??Yݵ?|г??*	??????K@2F
Iterator::ModelX9??v???!???pL@)???????1?Ṱ?HE@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM?O???!???]8?2@)?St$????1Wg?bnu.@:Preprocessing2U
Iterator::Model::ParallelMapV2?q?????!??ش?,@)?q?????1??ش?,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;?O??n??!9?߅?0@)???_vO~?1?6?'+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???Mb??!??.D?E@)F%u?k?1????]8@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??H?}]?!???d	l
@)??H?}]?1???d	l
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-C??6Z?!?w? z|@)-C??6Z?1?w? z|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ?????!?H????4@)/n??R?1Z???% @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 16.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t14.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?ma0l[0@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?f??j+???f??j+??!?f??j+??      ??!       "      ??!       *      ??!       2	??{??P????{??P??!??{??P??:      ??!       B      ??!       J	ݵ?|г??ݵ?|г??!ݵ?|г??R      ??!       Z	ݵ?|г??ݵ?|г??!ݵ?|г??JCPU_ONLYY?ma0l[0@b 