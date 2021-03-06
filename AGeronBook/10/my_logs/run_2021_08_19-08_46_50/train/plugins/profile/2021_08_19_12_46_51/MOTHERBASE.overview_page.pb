?	??|гi9@??|гi9@!??|гi9@	r?Z?k?@r?Z?k?@!r?Z?k?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??|гi9@?Zd;#@A?St$??@Y8gDio?@*	     I?@2F
Iterator::Model;pΈ??@!??????X@)f??a??@1?}????X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ZӼ???!???9M??)?o_???1?
"????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate9??v????!P?????){?G?z??1??Z????:Preprocessing2U
Iterator::Model::ParallelMapV2????Mb??!;
?H<???)????Mb??1;
?H<???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQ?|a2??!???????)-C??6z?1. ?t??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???_vOn?!v	?D???)???_vOn?1v	?D???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?h?!??6?-??)?~j?t?h?1??6?-??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?<,Ԛ???!U	??pU??)-C??6Z?1. ?t??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 31.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t37.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9s?Z?k?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Zd;#@?Zd;#@!?Zd;#@      ??!       "      ??!       *      ??!       2	?St$??@?St$??@!?St$??@:      ??!       B      ??!       J	8gDio?@8gDio?@!8gDio?@R      ??!       Z	8gDio?@8gDio?@!8gDio?@JCPU_ONLYYs?Z?k?@b Y      Y@qG?+O U@"?	
host?Your program is HIGHLY input-bound because 31.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t37.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?84.5048% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 