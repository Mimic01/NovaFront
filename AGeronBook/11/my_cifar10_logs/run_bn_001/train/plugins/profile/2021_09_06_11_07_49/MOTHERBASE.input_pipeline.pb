	?Fx?/@?Fx?/@!?Fx?/@	o??>?:??o??>?:??!o??>?:??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Fx?/@EGr?	!@A?ZӼ?T@Y??	h"??*	gffff?m@2F
Iterator::Model?`TR'???!"?2?0U@)??m4????1??S-pT@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????<,??!?pX?G? @)vq?-??1e????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?0?*??!~7 K^q@)???Q?~?1v?ح?f	@:Preprocessing2U
Iterator::Model::ParallelMapV2?ZӼ?}?!?G??#@)?ZӼ?}?1?G??#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;?O??n??!?7j?{.@)?J?4q?1??w?Qs??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?q????o?!*@?k??)?q????o?1*@?k??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceǺ???f?!
?OЋ???)Ǻ???f?1
?OЋ???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??@??ǈ?!?R?U?}@)??H?}]?1???T?b??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9o??>?:??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	EGr?	!@EGr?	!@!EGr?	!@      ??!       "      ??!       *      ??!       2	?ZӼ?T@?ZӼ?T@!?ZӼ?T@:      ??!       B      ??!       J	??	h"????	h"??!??	h"??R      ??!       Z	??	h"????	h"??!??	h"??JCPU_ONLYYo??>?:??b 