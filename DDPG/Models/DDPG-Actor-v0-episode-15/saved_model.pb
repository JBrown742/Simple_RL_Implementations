??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??

?
layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_6/gamma
?
/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma*
_output_shapes
:*
dtype0
?
layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_6/beta
?
.layer_normalization_6/beta/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:@*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:@*
dtype0
?
layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_7/gamma
?
/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma*
_output_shapes
:@*
dtype0
?
layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_7/beta
?
.layer_normalization_7/beta/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta*
_output_shapes
:@*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:@@*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:@*
dtype0
?
layer_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_8/gamma
?
/layer_normalization_8/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_8/gamma*
_output_shapes
:@*
dtype0
?
layer_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_8/beta
?
.layer_normalization_8/beta/Read/ReadVariableOpReadVariableOplayer_normalization_8/beta*
_output_shapes
:@*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:@*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
value? B?  B? 
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
q
axis
	gamma
beta
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
q
axis
	gamma
beta
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
q
(axis
	)gamma
*beta
+	variables
,trainable_variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
R
5	variables
6trainable_variables
7regularization_losses
8	keras_api
V
0
1
2
3
4
5
"6
#7
)8
*9
/10
011
V
0
1
2
3
4
5
"6
#7
)8
*9
/10
011
 
?
		variables
9layer_regularization_losses
:metrics
;layer_metrics

trainable_variables
<non_trainable_variables

=layers
regularization_losses
 
 
fd
VARIABLE_VALUElayer_normalization_6/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_6/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
>layer_regularization_losses
?metrics
@layer_metrics
trainable_variables
Anon_trainable_variables

Blayers
regularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
Clayer_regularization_losses
Dmetrics
Elayer_metrics
trainable_variables
Fnon_trainable_variables

Glayers
regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_7/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_7/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
Hlayer_regularization_losses
Imetrics
Jlayer_metrics
trainable_variables
Knon_trainable_variables

Llayers
 regularization_losses
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
?
$	variables
Mlayer_regularization_losses
Nmetrics
Olayer_metrics
%trainable_variables
Pnon_trainable_variables

Qlayers
&regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_8/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_8/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
+	variables
Rlayer_regularization_losses
Smetrics
Tlayer_metrics
,trainable_variables
Unon_trainable_variables

Vlayers
-regularization_losses
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
1	variables
Wlayer_regularization_losses
Xmetrics
Ylayer_metrics
2trainable_variables
Znon_trainable_variables

[layers
3regularization_losses
 
 
 
?
5	variables
\layer_regularization_losses
]metrics
^layer_metrics
6trainable_variables
_non_trainable_variables

`layers
7regularization_losses
 
 
 
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3layer_normalization_6/gammalayer_normalization_6/betadense_6/kerneldense_6/biaslayer_normalization_7/gammalayer_normalization_7/betadense_7/kerneldense_7/biaslayer_normalization_8/gammalayer_normalization_8/betadense_8/kerneldense_8/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_4102215
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/layer_normalization_6/gamma/Read/ReadVariableOp.layer_normalization_6/beta/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp/layer_normalization_7/gamma/Read/ReadVariableOp.layer_normalization_7/beta/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp/layer_normalization_8/gamma/Read/ReadVariableOp.layer_normalization_8/beta/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_4102880
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization_6/gammalayer_normalization_6/betadense_6/kerneldense_6/biaslayer_normalization_7/gammalayer_normalization_7/betadense_7/kerneldense_7/biaslayer_normalization_8/gammalayer_normalization_8/betadense_8/kerneldense_8/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_4102926ۥ

?(
?
D__inference_model_2_layer_call_and_return_conditional_losses_4101899

inputs+
layer_normalization_6_4101728:+
layer_normalization_6_4101730:!
dense_6_4101745:@
dense_6_4101747:@+
layer_normalization_7_4101797:@+
layer_normalization_7_4101799:@!
dense_7_4101814:@@
dense_7_4101816:@+
layer_normalization_8_4101866:@+
layer_normalization_8_4101868:@!
dense_8_4101883:@
dense_8_4101885:
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?-layer_normalization_6/StatefulPartitionedCall?-layer_normalization_7/StatefulPartitionedCall?-layer_normalization_8/StatefulPartitionedCall?
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_6_4101728layer_normalization_6_4101730*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_6_layer_call_and_return_conditional_losses_41017272/
-layer_normalization_6/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0dense_6_4101745dense_6_4101747*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_41017442!
dense_6/StatefulPartitionedCall?
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0layer_normalization_7_4101797layer_normalization_7_4101799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_7_layer_call_and_return_conditional_losses_41017962/
-layer_normalization_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0dense_7_4101814dense_7_4101816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_41018132!
dense_7/StatefulPartitionedCall?
-layer_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0layer_normalization_8_4101866layer_normalization_8_4101868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_8_layer_call_and_return_conditional_losses_41018652/
-layer_normalization_8/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_8/StatefulPartitionedCall:output:0dense_8_4101883dense_8_4101885*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_41018822!
dense_8/StatefulPartitionedCall?
rescaling_1/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_41018962
rescaling_1/PartitionedCall
IdentityIdentity$rescaling_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall.^layer_normalization_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2^
-layer_normalization_8/StatefulPartitionedCall-layer_normalization_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_4101674
input_3I
;model_2_layer_normalization_6_mul_2_readvariableop_resource:G
9model_2_layer_normalization_6_add_readvariableop_resource:@
.model_2_dense_6_matmul_readvariableop_resource:@=
/model_2_dense_6_biasadd_readvariableop_resource:@I
;model_2_layer_normalization_7_mul_2_readvariableop_resource:@G
9model_2_layer_normalization_7_add_readvariableop_resource:@@
.model_2_dense_7_matmul_readvariableop_resource:@@=
/model_2_dense_7_biasadd_readvariableop_resource:@I
;model_2_layer_normalization_8_mul_2_readvariableop_resource:@G
9model_2_layer_normalization_8_add_readvariableop_resource:@@
.model_2_dense_8_matmul_readvariableop_resource:@=
/model_2_dense_8_biasadd_readvariableop_resource:
identity??&model_2/dense_6/BiasAdd/ReadVariableOp?%model_2/dense_6/MatMul/ReadVariableOp?&model_2/dense_7/BiasAdd/ReadVariableOp?%model_2/dense_7/MatMul/ReadVariableOp?&model_2/dense_8/BiasAdd/ReadVariableOp?%model_2/dense_8/MatMul/ReadVariableOp?0model_2/layer_normalization_6/add/ReadVariableOp?2model_2/layer_normalization_6/mul_2/ReadVariableOp?0model_2/layer_normalization_7/add/ReadVariableOp?2model_2/layer_normalization_7/mul_2/ReadVariableOp?0model_2/layer_normalization_8/add/ReadVariableOp?2model_2/layer_normalization_8/mul_2/ReadVariableOp?
#model_2/layer_normalization_6/ShapeShapeinput_3*
T0*
_output_shapes
:2%
#model_2/layer_normalization_6/Shape?
1model_2/layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_2/layer_normalization_6/strided_slice/stack?
3model_2/layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_2/layer_normalization_6/strided_slice/stack_1?
3model_2/layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_2/layer_normalization_6/strided_slice/stack_2?
+model_2/layer_normalization_6/strided_sliceStridedSlice,model_2/layer_normalization_6/Shape:output:0:model_2/layer_normalization_6/strided_slice/stack:output:0<model_2/layer_normalization_6/strided_slice/stack_1:output:0<model_2/layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_2/layer_normalization_6/strided_slice?
#model_2/layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_2/layer_normalization_6/mul/x?
!model_2/layer_normalization_6/mulMul,model_2/layer_normalization_6/mul/x:output:04model_2/layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: 2#
!model_2/layer_normalization_6/mul?
3model_2/layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3model_2/layer_normalization_6/strided_slice_1/stack?
5model_2/layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5model_2/layer_normalization_6/strided_slice_1/stack_1?
5model_2/layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5model_2/layer_normalization_6/strided_slice_1/stack_2?
-model_2/layer_normalization_6/strided_slice_1StridedSlice,model_2/layer_normalization_6/Shape:output:0<model_2/layer_normalization_6/strided_slice_1/stack:output:0>model_2/layer_normalization_6/strided_slice_1/stack_1:output:0>model_2/layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-model_2/layer_normalization_6/strided_slice_1?
%model_2/layer_normalization_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_2/layer_normalization_6/mul_1/x?
#model_2/layer_normalization_6/mul_1Mul.model_2/layer_normalization_6/mul_1/x:output:06model_2/layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: 2%
#model_2/layer_normalization_6/mul_1?
-model_2/layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_2/layer_normalization_6/Reshape/shape/0?
-model_2/layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_2/layer_normalization_6/Reshape/shape/3?
+model_2/layer_normalization_6/Reshape/shapePack6model_2/layer_normalization_6/Reshape/shape/0:output:0%model_2/layer_normalization_6/mul:z:0'model_2/layer_normalization_6/mul_1:z:06model_2/layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+model_2/layer_normalization_6/Reshape/shape?
%model_2/layer_normalization_6/ReshapeReshapeinput_34model_2/layer_normalization_6/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%model_2/layer_normalization_6/Reshape?
)model_2/layer_normalization_6/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2+
)model_2/layer_normalization_6/ones/Less/y?
'model_2/layer_normalization_6/ones/LessLess%model_2/layer_normalization_6/mul:z:02model_2/layer_normalization_6/ones/Less/y:output:0*
T0*
_output_shapes
: 2)
'model_2/layer_normalization_6/ones/Less?
)model_2/layer_normalization_6/ones/packedPack%model_2/layer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2+
)model_2/layer_normalization_6/ones/packed?
(model_2/layer_normalization_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(model_2/layer_normalization_6/ones/Const?
"model_2/layer_normalization_6/onesFill2model_2/layer_normalization_6/ones/packed:output:01model_2/layer_normalization_6/ones/Const:output:0*
T0*#
_output_shapes
:?????????2$
"model_2/layer_normalization_6/ones?
*model_2/layer_normalization_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2,
*model_2/layer_normalization_6/zeros/Less/y?
(model_2/layer_normalization_6/zeros/LessLess%model_2/layer_normalization_6/mul:z:03model_2/layer_normalization_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2*
(model_2/layer_normalization_6/zeros/Less?
*model_2/layer_normalization_6/zeros/packedPack%model_2/layer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2,
*model_2/layer_normalization_6/zeros/packed?
)model_2/layer_normalization_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)model_2/layer_normalization_6/zeros/Const?
#model_2/layer_normalization_6/zerosFill3model_2/layer_normalization_6/zeros/packed:output:02model_2/layer_normalization_6/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2%
#model_2/layer_normalization_6/zeros?
#model_2/layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB 2%
#model_2/layer_normalization_6/Const?
%model_2/layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2'
%model_2/layer_normalization_6/Const_1?
.model_2/layer_normalization_6/FusedBatchNormV3FusedBatchNormV3.model_2/layer_normalization_6/Reshape:output:0+model_2/layer_normalization_6/ones:output:0,model_2/layer_normalization_6/zeros:output:0,model_2/layer_normalization_6/Const:output:0.model_2/layer_normalization_6/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:20
.model_2/layer_normalization_6/FusedBatchNormV3?
'model_2/layer_normalization_6/Reshape_1Reshape2model_2/layer_normalization_6/FusedBatchNormV3:y:0,model_2/layer_normalization_6/Shape:output:0*
T0*'
_output_shapes
:?????????2)
'model_2/layer_normalization_6/Reshape_1?
2model_2/layer_normalization_6/mul_2/ReadVariableOpReadVariableOp;model_2_layer_normalization_6_mul_2_readvariableop_resource*
_output_shapes
:*
dtype024
2model_2/layer_normalization_6/mul_2/ReadVariableOp?
#model_2/layer_normalization_6/mul_2Mul0model_2/layer_normalization_6/Reshape_1:output:0:model_2/layer_normalization_6/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#model_2/layer_normalization_6/mul_2?
0model_2/layer_normalization_6/add/ReadVariableOpReadVariableOp9model_2_layer_normalization_6_add_readvariableop_resource*
_output_shapes
:*
dtype022
0model_2/layer_normalization_6/add/ReadVariableOp?
!model_2/layer_normalization_6/addAddV2'model_2/layer_normalization_6/mul_2:z:08model_2/layer_normalization_6/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!model_2/layer_normalization_6/add?
%model_2/dense_6/MatMul/ReadVariableOpReadVariableOp.model_2_dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%model_2/dense_6/MatMul/ReadVariableOp?
model_2/dense_6/MatMulMatMul%model_2/layer_normalization_6/add:z:0-model_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_6/MatMul?
&model_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&model_2/dense_6/BiasAdd/ReadVariableOp?
model_2/dense_6/BiasAddBiasAdd model_2/dense_6/MatMul:product:0.model_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_6/BiasAdd?
model_2/dense_6/ReluRelu model_2/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_6/Relu?
#model_2/layer_normalization_7/ShapeShape"model_2/dense_6/Relu:activations:0*
T0*
_output_shapes
:2%
#model_2/layer_normalization_7/Shape?
1model_2/layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_2/layer_normalization_7/strided_slice/stack?
3model_2/layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_2/layer_normalization_7/strided_slice/stack_1?
3model_2/layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_2/layer_normalization_7/strided_slice/stack_2?
+model_2/layer_normalization_7/strided_sliceStridedSlice,model_2/layer_normalization_7/Shape:output:0:model_2/layer_normalization_7/strided_slice/stack:output:0<model_2/layer_normalization_7/strided_slice/stack_1:output:0<model_2/layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_2/layer_normalization_7/strided_slice?
#model_2/layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_2/layer_normalization_7/mul/x?
!model_2/layer_normalization_7/mulMul,model_2/layer_normalization_7/mul/x:output:04model_2/layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: 2#
!model_2/layer_normalization_7/mul?
3model_2/layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3model_2/layer_normalization_7/strided_slice_1/stack?
5model_2/layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5model_2/layer_normalization_7/strided_slice_1/stack_1?
5model_2/layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5model_2/layer_normalization_7/strided_slice_1/stack_2?
-model_2/layer_normalization_7/strided_slice_1StridedSlice,model_2/layer_normalization_7/Shape:output:0<model_2/layer_normalization_7/strided_slice_1/stack:output:0>model_2/layer_normalization_7/strided_slice_1/stack_1:output:0>model_2/layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-model_2/layer_normalization_7/strided_slice_1?
%model_2/layer_normalization_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_2/layer_normalization_7/mul_1/x?
#model_2/layer_normalization_7/mul_1Mul.model_2/layer_normalization_7/mul_1/x:output:06model_2/layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: 2%
#model_2/layer_normalization_7/mul_1?
-model_2/layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_2/layer_normalization_7/Reshape/shape/0?
-model_2/layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_2/layer_normalization_7/Reshape/shape/3?
+model_2/layer_normalization_7/Reshape/shapePack6model_2/layer_normalization_7/Reshape/shape/0:output:0%model_2/layer_normalization_7/mul:z:0'model_2/layer_normalization_7/mul_1:z:06model_2/layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+model_2/layer_normalization_7/Reshape/shape?
%model_2/layer_normalization_7/ReshapeReshape"model_2/dense_6/Relu:activations:04model_2/layer_normalization_7/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%model_2/layer_normalization_7/Reshape?
)model_2/layer_normalization_7/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2+
)model_2/layer_normalization_7/ones/Less/y?
'model_2/layer_normalization_7/ones/LessLess%model_2/layer_normalization_7/mul:z:02model_2/layer_normalization_7/ones/Less/y:output:0*
T0*
_output_shapes
: 2)
'model_2/layer_normalization_7/ones/Less?
)model_2/layer_normalization_7/ones/packedPack%model_2/layer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2+
)model_2/layer_normalization_7/ones/packed?
(model_2/layer_normalization_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(model_2/layer_normalization_7/ones/Const?
"model_2/layer_normalization_7/onesFill2model_2/layer_normalization_7/ones/packed:output:01model_2/layer_normalization_7/ones/Const:output:0*
T0*#
_output_shapes
:?????????2$
"model_2/layer_normalization_7/ones?
*model_2/layer_normalization_7/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2,
*model_2/layer_normalization_7/zeros/Less/y?
(model_2/layer_normalization_7/zeros/LessLess%model_2/layer_normalization_7/mul:z:03model_2/layer_normalization_7/zeros/Less/y:output:0*
T0*
_output_shapes
: 2*
(model_2/layer_normalization_7/zeros/Less?
*model_2/layer_normalization_7/zeros/packedPack%model_2/layer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2,
*model_2/layer_normalization_7/zeros/packed?
)model_2/layer_normalization_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)model_2/layer_normalization_7/zeros/Const?
#model_2/layer_normalization_7/zerosFill3model_2/layer_normalization_7/zeros/packed:output:02model_2/layer_normalization_7/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2%
#model_2/layer_normalization_7/zeros?
#model_2/layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB 2%
#model_2/layer_normalization_7/Const?
%model_2/layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2'
%model_2/layer_normalization_7/Const_1?
.model_2/layer_normalization_7/FusedBatchNormV3FusedBatchNormV3.model_2/layer_normalization_7/Reshape:output:0+model_2/layer_normalization_7/ones:output:0,model_2/layer_normalization_7/zeros:output:0,model_2/layer_normalization_7/Const:output:0.model_2/layer_normalization_7/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:20
.model_2/layer_normalization_7/FusedBatchNormV3?
'model_2/layer_normalization_7/Reshape_1Reshape2model_2/layer_normalization_7/FusedBatchNormV3:y:0,model_2/layer_normalization_7/Shape:output:0*
T0*'
_output_shapes
:?????????@2)
'model_2/layer_normalization_7/Reshape_1?
2model_2/layer_normalization_7/mul_2/ReadVariableOpReadVariableOp;model_2_layer_normalization_7_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype024
2model_2/layer_normalization_7/mul_2/ReadVariableOp?
#model_2/layer_normalization_7/mul_2Mul0model_2/layer_normalization_7/Reshape_1:output:0:model_2/layer_normalization_7/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2%
#model_2/layer_normalization_7/mul_2?
0model_2/layer_normalization_7/add/ReadVariableOpReadVariableOp9model_2_layer_normalization_7_add_readvariableop_resource*
_output_shapes
:@*
dtype022
0model_2/layer_normalization_7/add/ReadVariableOp?
!model_2/layer_normalization_7/addAddV2'model_2/layer_normalization_7/mul_2:z:08model_2/layer_normalization_7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2#
!model_2/layer_normalization_7/add?
%model_2/dense_7/MatMul/ReadVariableOpReadVariableOp.model_2_dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%model_2/dense_7/MatMul/ReadVariableOp?
model_2/dense_7/MatMulMatMul%model_2/layer_normalization_7/add:z:0-model_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_7/MatMul?
&model_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&model_2/dense_7/BiasAdd/ReadVariableOp?
model_2/dense_7/BiasAddBiasAdd model_2/dense_7/MatMul:product:0.model_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_7/BiasAdd?
model_2/dense_7/ReluRelu model_2/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_7/Relu?
#model_2/layer_normalization_8/ShapeShape"model_2/dense_7/Relu:activations:0*
T0*
_output_shapes
:2%
#model_2/layer_normalization_8/Shape?
1model_2/layer_normalization_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_2/layer_normalization_8/strided_slice/stack?
3model_2/layer_normalization_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_2/layer_normalization_8/strided_slice/stack_1?
3model_2/layer_normalization_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_2/layer_normalization_8/strided_slice/stack_2?
+model_2/layer_normalization_8/strided_sliceStridedSlice,model_2/layer_normalization_8/Shape:output:0:model_2/layer_normalization_8/strided_slice/stack:output:0<model_2/layer_normalization_8/strided_slice/stack_1:output:0<model_2/layer_normalization_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_2/layer_normalization_8/strided_slice?
#model_2/layer_normalization_8/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_2/layer_normalization_8/mul/x?
!model_2/layer_normalization_8/mulMul,model_2/layer_normalization_8/mul/x:output:04model_2/layer_normalization_8/strided_slice:output:0*
T0*
_output_shapes
: 2#
!model_2/layer_normalization_8/mul?
3model_2/layer_normalization_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3model_2/layer_normalization_8/strided_slice_1/stack?
5model_2/layer_normalization_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5model_2/layer_normalization_8/strided_slice_1/stack_1?
5model_2/layer_normalization_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5model_2/layer_normalization_8/strided_slice_1/stack_2?
-model_2/layer_normalization_8/strided_slice_1StridedSlice,model_2/layer_normalization_8/Shape:output:0<model_2/layer_normalization_8/strided_slice_1/stack:output:0>model_2/layer_normalization_8/strided_slice_1/stack_1:output:0>model_2/layer_normalization_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-model_2/layer_normalization_8/strided_slice_1?
%model_2/layer_normalization_8/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_2/layer_normalization_8/mul_1/x?
#model_2/layer_normalization_8/mul_1Mul.model_2/layer_normalization_8/mul_1/x:output:06model_2/layer_normalization_8/strided_slice_1:output:0*
T0*
_output_shapes
: 2%
#model_2/layer_normalization_8/mul_1?
-model_2/layer_normalization_8/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_2/layer_normalization_8/Reshape/shape/0?
-model_2/layer_normalization_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_2/layer_normalization_8/Reshape/shape/3?
+model_2/layer_normalization_8/Reshape/shapePack6model_2/layer_normalization_8/Reshape/shape/0:output:0%model_2/layer_normalization_8/mul:z:0'model_2/layer_normalization_8/mul_1:z:06model_2/layer_normalization_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+model_2/layer_normalization_8/Reshape/shape?
%model_2/layer_normalization_8/ReshapeReshape"model_2/dense_7/Relu:activations:04model_2/layer_normalization_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%model_2/layer_normalization_8/Reshape?
)model_2/layer_normalization_8/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2+
)model_2/layer_normalization_8/ones/Less/y?
'model_2/layer_normalization_8/ones/LessLess%model_2/layer_normalization_8/mul:z:02model_2/layer_normalization_8/ones/Less/y:output:0*
T0*
_output_shapes
: 2)
'model_2/layer_normalization_8/ones/Less?
)model_2/layer_normalization_8/ones/packedPack%model_2/layer_normalization_8/mul:z:0*
N*
T0*
_output_shapes
:2+
)model_2/layer_normalization_8/ones/packed?
(model_2/layer_normalization_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(model_2/layer_normalization_8/ones/Const?
"model_2/layer_normalization_8/onesFill2model_2/layer_normalization_8/ones/packed:output:01model_2/layer_normalization_8/ones/Const:output:0*
T0*#
_output_shapes
:?????????2$
"model_2/layer_normalization_8/ones?
*model_2/layer_normalization_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2,
*model_2/layer_normalization_8/zeros/Less/y?
(model_2/layer_normalization_8/zeros/LessLess%model_2/layer_normalization_8/mul:z:03model_2/layer_normalization_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2*
(model_2/layer_normalization_8/zeros/Less?
*model_2/layer_normalization_8/zeros/packedPack%model_2/layer_normalization_8/mul:z:0*
N*
T0*
_output_shapes
:2,
*model_2/layer_normalization_8/zeros/packed?
)model_2/layer_normalization_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)model_2/layer_normalization_8/zeros/Const?
#model_2/layer_normalization_8/zerosFill3model_2/layer_normalization_8/zeros/packed:output:02model_2/layer_normalization_8/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2%
#model_2/layer_normalization_8/zeros?
#model_2/layer_normalization_8/ConstConst*
_output_shapes
: *
dtype0*
valueB 2%
#model_2/layer_normalization_8/Const?
%model_2/layer_normalization_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2'
%model_2/layer_normalization_8/Const_1?
.model_2/layer_normalization_8/FusedBatchNormV3FusedBatchNormV3.model_2/layer_normalization_8/Reshape:output:0+model_2/layer_normalization_8/ones:output:0,model_2/layer_normalization_8/zeros:output:0,model_2/layer_normalization_8/Const:output:0.model_2/layer_normalization_8/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:20
.model_2/layer_normalization_8/FusedBatchNormV3?
'model_2/layer_normalization_8/Reshape_1Reshape2model_2/layer_normalization_8/FusedBatchNormV3:y:0,model_2/layer_normalization_8/Shape:output:0*
T0*'
_output_shapes
:?????????@2)
'model_2/layer_normalization_8/Reshape_1?
2model_2/layer_normalization_8/mul_2/ReadVariableOpReadVariableOp;model_2_layer_normalization_8_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype024
2model_2/layer_normalization_8/mul_2/ReadVariableOp?
#model_2/layer_normalization_8/mul_2Mul0model_2/layer_normalization_8/Reshape_1:output:0:model_2/layer_normalization_8/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2%
#model_2/layer_normalization_8/mul_2?
0model_2/layer_normalization_8/add/ReadVariableOpReadVariableOp9model_2_layer_normalization_8_add_readvariableop_resource*
_output_shapes
:@*
dtype022
0model_2/layer_normalization_8/add/ReadVariableOp?
!model_2/layer_normalization_8/addAddV2'model_2/layer_normalization_8/mul_2:z:08model_2/layer_normalization_8/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2#
!model_2/layer_normalization_8/add?
%model_2/dense_8/MatMul/ReadVariableOpReadVariableOp.model_2_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%model_2/dense_8/MatMul/ReadVariableOp?
model_2/dense_8/MatMulMatMul%model_2/layer_normalization_8/add:z:0-model_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/dense_8/MatMul?
&model_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_2/dense_8/BiasAdd/ReadVariableOp?
model_2/dense_8/BiasAddBiasAdd model_2/dense_8/MatMul:product:0.model_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/dense_8/BiasAdd?
model_2/dense_8/TanhTanh model_2/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/dense_8/Tanh}
model_2/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_2/rescaling_1/Cast/x?
model_2/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_2/rescaling_1/Cast_1/x?
model_2/rescaling_1/mulMulmodel_2/dense_8/Tanh:y:0#model_2/rescaling_1/Cast/x:output:0*
T0*'
_output_shapes
:?????????2
model_2/rescaling_1/mul?
model_2/rescaling_1/addAddV2model_2/rescaling_1/mul:z:0%model_2/rescaling_1/Cast_1/x:output:0*
T0*'
_output_shapes
:?????????2
model_2/rescaling_1/addv
IdentityIdentitymodel_2/rescaling_1/add:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp'^model_2/dense_6/BiasAdd/ReadVariableOp&^model_2/dense_6/MatMul/ReadVariableOp'^model_2/dense_7/BiasAdd/ReadVariableOp&^model_2/dense_7/MatMul/ReadVariableOp'^model_2/dense_8/BiasAdd/ReadVariableOp&^model_2/dense_8/MatMul/ReadVariableOp1^model_2/layer_normalization_6/add/ReadVariableOp3^model_2/layer_normalization_6/mul_2/ReadVariableOp1^model_2/layer_normalization_7/add/ReadVariableOp3^model_2/layer_normalization_7/mul_2/ReadVariableOp1^model_2/layer_normalization_8/add/ReadVariableOp3^model_2/layer_normalization_8/mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2P
&model_2/dense_6/BiasAdd/ReadVariableOp&model_2/dense_6/BiasAdd/ReadVariableOp2N
%model_2/dense_6/MatMul/ReadVariableOp%model_2/dense_6/MatMul/ReadVariableOp2P
&model_2/dense_7/BiasAdd/ReadVariableOp&model_2/dense_7/BiasAdd/ReadVariableOp2N
%model_2/dense_7/MatMul/ReadVariableOp%model_2/dense_7/MatMul/ReadVariableOp2P
&model_2/dense_8/BiasAdd/ReadVariableOp&model_2/dense_8/BiasAdd/ReadVariableOp2N
%model_2/dense_8/MatMul/ReadVariableOp%model_2/dense_8/MatMul/ReadVariableOp2d
0model_2/layer_normalization_6/add/ReadVariableOp0model_2/layer_normalization_6/add/ReadVariableOp2h
2model_2/layer_normalization_6/mul_2/ReadVariableOp2model_2/layer_normalization_6/mul_2/ReadVariableOp2d
0model_2/layer_normalization_7/add/ReadVariableOp0model_2/layer_normalization_7/add/ReadVariableOp2h
2model_2/layer_normalization_7/mul_2/ReadVariableOp2model_2/layer_normalization_7/mul_2/ReadVariableOp2d
0model_2/layer_normalization_8/add/ReadVariableOp0model_2/layer_normalization_8/add/ReadVariableOp2h
2model_2/layer_normalization_8/mul_2/ReadVariableOp2model_2/layer_normalization_8/mul_2/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?
d
H__inference_rescaling_1_layer_call_and_return_conditional_losses_4102821

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Cast/xY
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

Cast_1/x\
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:?????????2
mula
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_7_layer_call_fn_4102722

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_41018132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?7
?
#__inference__traced_restore_4102926
file_prefix:
,assignvariableop_layer_normalization_6_gamma:;
-assignvariableop_1_layer_normalization_6_beta:3
!assignvariableop_2_dense_6_kernel:@-
assignvariableop_3_dense_6_bias:@<
.assignvariableop_4_layer_normalization_7_gamma:@;
-assignvariableop_5_layer_normalization_7_beta:@3
!assignvariableop_6_dense_7_kernel:@@-
assignvariableop_7_dense_7_bias:@<
.assignvariableop_8_layer_normalization_8_gamma:@;
-assignvariableop_9_layer_normalization_8_beta:@4
"assignvariableop_10_dense_8_kernel:@.
 assignvariableop_11_dense_8_bias:
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp,assignvariableop_layer_normalization_6_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp-assignvariableop_1_layer_normalization_6_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_6_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_6_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_layer_normalization_7_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp-assignvariableop_5_layer_normalization_7_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_layer_normalization_8_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_layer_normalization_8_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_8_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_8_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12f
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_13?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
)__inference_model_2_layer_call_fn_4102273

inputs
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_41020582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_rescaling_1_layer_call_and_return_conditional_losses_4101896

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Cast/xY
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

Cast_1/x\
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:?????????2
mula
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
R__inference_layer_normalization_6_layer_call_and_return_conditional_losses_4101727

inputs+
mul_2_readvariableop_resource:)
add_readvariableop_resource:
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
Reshape]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/y^
	ones/LessLessmul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/LessY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????2
ones_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/ya

zeros/LessLessmul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Less[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_dense_8_layer_call_and_return_conditional_losses_4101882

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_model_2_layer_call_fn_4102114
input_3
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_41020582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
D__inference_dense_6_layer_call_and_return_conditional_losses_4102658

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_6_layer_call_fn_4102647

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_41017442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
R__inference_layer_normalization_8_layer_call_and_return_conditional_losses_4101865

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
Reshape]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/y^
	ones/LessLessmul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/LessY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????2
ones_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/ya

zeros/LessLessmul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Less[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_dense_7_layer_call_and_return_conditional_losses_4102733

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?(
?
D__inference_model_2_layer_call_and_return_conditional_losses_4102058

inputs+
layer_normalization_6_4102026:+
layer_normalization_6_4102028:!
dense_6_4102031:@
dense_6_4102033:@+
layer_normalization_7_4102036:@+
layer_normalization_7_4102038:@!
dense_7_4102041:@@
dense_7_4102043:@+
layer_normalization_8_4102046:@+
layer_normalization_8_4102048:@!
dense_8_4102051:@
dense_8_4102053:
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?-layer_normalization_6/StatefulPartitionedCall?-layer_normalization_7/StatefulPartitionedCall?-layer_normalization_8/StatefulPartitionedCall?
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_6_4102026layer_normalization_6_4102028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_6_layer_call_and_return_conditional_losses_41017272/
-layer_normalization_6/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0dense_6_4102031dense_6_4102033*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_41017442!
dense_6/StatefulPartitionedCall?
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0layer_normalization_7_4102036layer_normalization_7_4102038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_7_layer_call_and_return_conditional_losses_41017962/
-layer_normalization_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0dense_7_4102041dense_7_4102043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_41018132!
dense_7/StatefulPartitionedCall?
-layer_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0layer_normalization_8_4102046layer_normalization_8_4102048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_8_layer_call_and_return_conditional_losses_41018652/
-layer_normalization_8/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_8/StatefulPartitionedCall:output:0dense_8_4102051dense_8_4102053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_41018822!
dense_8/StatefulPartitionedCall?
rescaling_1/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_41018962
rescaling_1/PartitionedCall
IdentityIdentity$rescaling_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall.^layer_normalization_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2^
-layer_normalization_8/StatefulPartitionedCall-layer_normalization_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_layer_normalization_6_layer_call_fn_4102592

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_6_layer_call_and_return_conditional_losses_41017272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
-__inference_rescaling_1_layer_call_fn_4102813

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_41018962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
R__inference_layer_normalization_7_layer_call_and_return_conditional_losses_4101796

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
Reshape]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/y^
	ones/LessLessmul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/LessY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????2
ones_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/ya

zeros/LessLessmul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Less[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?(
?
D__inference_model_2_layer_call_and_return_conditional_losses_4102184
input_3+
layer_normalization_6_4102152:+
layer_normalization_6_4102154:!
dense_6_4102157:@
dense_6_4102159:@+
layer_normalization_7_4102162:@+
layer_normalization_7_4102164:@!
dense_7_4102167:@@
dense_7_4102169:@+
layer_normalization_8_4102172:@+
layer_normalization_8_4102174:@!
dense_8_4102177:@
dense_8_4102179:
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?-layer_normalization_6/StatefulPartitionedCall?-layer_normalization_7/StatefulPartitionedCall?-layer_normalization_8/StatefulPartitionedCall?
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCallinput_3layer_normalization_6_4102152layer_normalization_6_4102154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_6_layer_call_and_return_conditional_losses_41017272/
-layer_normalization_6/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0dense_6_4102157dense_6_4102159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_41017442!
dense_6/StatefulPartitionedCall?
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0layer_normalization_7_4102162layer_normalization_7_4102164*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_7_layer_call_and_return_conditional_losses_41017962/
-layer_normalization_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0dense_7_4102167dense_7_4102169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_41018132!
dense_7/StatefulPartitionedCall?
-layer_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0layer_normalization_8_4102172layer_normalization_8_4102174*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_8_layer_call_and_return_conditional_losses_41018652/
-layer_normalization_8/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_8/StatefulPartitionedCall:output:0dense_8_4102177dense_8_4102179*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_41018822!
dense_8/StatefulPartitionedCall?
rescaling_1/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_41018962
rescaling_1/PartitionedCall
IdentityIdentity$rescaling_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall.^layer_normalization_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2^
-layer_normalization_8/StatefulPartitionedCall-layer_normalization_8/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?'
?
R__inference_layer_normalization_7_layer_call_and_return_conditional_losses_4102713

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
Reshape]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/y^
	ones/LessLessmul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/LessY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????2
ones_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/ya

zeros/LessLessmul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Less[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_dense_8_layer_call_and_return_conditional_losses_4102808

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?&
?
 __inference__traced_save_4102880
file_prefix:
6savev2_layer_normalization_6_gamma_read_readvariableop9
5savev2_layer_normalization_6_beta_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop:
6savev2_layer_normalization_7_gamma_read_readvariableop9
5savev2_layer_normalization_7_beta_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop:
6savev2_layer_normalization_8_gamma_read_readvariableop9
5savev2_layer_normalization_8_beta_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_layer_normalization_6_gamma_read_readvariableop5savev2_layer_normalization_6_beta_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop6savev2_layer_normalization_7_gamma_read_readvariableop5savev2_layer_normalization_7_beta_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop6savev2_layer_normalization_8_gamma_read_readvariableop5savev2_layer_normalization_8_beta_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*k
_input_shapesZ
X: :::@:@:@:@:@@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
?'
?
R__inference_layer_normalization_6_layer_call_and_return_conditional_losses_4102638

inputs+
mul_2_readvariableop_resource:)
add_readvariableop_resource:
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
Reshape]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/y^
	ones/LessLessmul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/LessY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????2
ones_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/ya

zeros/LessLessmul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Less[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_layer_normalization_8_layer_call_fn_4102742

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_8_layer_call_and_return_conditional_losses_41018652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?(
?
D__inference_model_2_layer_call_and_return_conditional_losses_4102149
input_3+
layer_normalization_6_4102117:+
layer_normalization_6_4102119:!
dense_6_4102122:@
dense_6_4102124:@+
layer_normalization_7_4102127:@+
layer_normalization_7_4102129:@!
dense_7_4102132:@@
dense_7_4102134:@+
layer_normalization_8_4102137:@+
layer_normalization_8_4102139:@!
dense_8_4102142:@
dense_8_4102144:
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?-layer_normalization_6/StatefulPartitionedCall?-layer_normalization_7/StatefulPartitionedCall?-layer_normalization_8/StatefulPartitionedCall?
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCallinput_3layer_normalization_6_4102117layer_normalization_6_4102119*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_6_layer_call_and_return_conditional_losses_41017272/
-layer_normalization_6/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0dense_6_4102122dense_6_4102124*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_41017442!
dense_6/StatefulPartitionedCall?
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0layer_normalization_7_4102127layer_normalization_7_4102129*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_7_layer_call_and_return_conditional_losses_41017962/
-layer_normalization_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0dense_7_4102132dense_7_4102134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_41018132!
dense_7/StatefulPartitionedCall?
-layer_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0layer_normalization_8_4102137layer_normalization_8_4102139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_8_layer_call_and_return_conditional_losses_41018652/
-layer_normalization_8/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_8/StatefulPartitionedCall:output:0dense_8_4102142dense_8_4102144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_41018822!
dense_8/StatefulPartitionedCall?
rescaling_1/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_41018962
rescaling_1/PartitionedCall
IdentityIdentity$rescaling_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall.^layer_normalization_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2^
-layer_normalization_8/StatefulPartitionedCall-layer_normalization_8/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
??
?

D__inference_model_2_layer_call_and_return_conditional_losses_4102583

inputsA
3layer_normalization_6_mul_2_readvariableop_resource:?
1layer_normalization_6_add_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:@5
'dense_6_biasadd_readvariableop_resource:@A
3layer_normalization_7_mul_2_readvariableop_resource:@?
1layer_normalization_7_add_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@A
3layer_normalization_8_mul_2_readvariableop_resource:@?
1layer_normalization_8_add_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@5
'dense_8_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?(layer_normalization_6/add/ReadVariableOp?*layer_normalization_6/mul_2/ReadVariableOp?(layer_normalization_7/add/ReadVariableOp?*layer_normalization_7/mul_2/ReadVariableOp?(layer_normalization_8/add/ReadVariableOp?*layer_normalization_8/mul_2/ReadVariableOpp
layer_normalization_6/ShapeShapeinputs*
T0*
_output_shapes
:2
layer_normalization_6/Shape?
)layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_6/strided_slice/stack?
+layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice/stack_1?
+layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice/stack_2?
#layer_normalization_6/strided_sliceStridedSlice$layer_normalization_6/Shape:output:02layer_normalization_6/strided_slice/stack:output:04layer_normalization_6/strided_slice/stack_1:output:04layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_6/strided_slice|
layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_6/mul/x?
layer_normalization_6/mulMul$layer_normalization_6/mul/x:output:0,layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_6/mul?
+layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice_1/stack?
-layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_6/strided_slice_1/stack_1?
-layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_6/strided_slice_1/stack_2?
%layer_normalization_6/strided_slice_1StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_1/stack:output:06layer_normalization_6/strided_slice_1/stack_1:output:06layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_6/strided_slice_1?
layer_normalization_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_6/mul_1/x?
layer_normalization_6/mul_1Mul&layer_normalization_6/mul_1/x:output:0.layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_6/mul_1?
%layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_6/Reshape/shape/0?
%layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_6/Reshape/shape/3?
#layer_normalization_6/Reshape/shapePack.layer_normalization_6/Reshape/shape/0:output:0layer_normalization_6/mul:z:0layer_normalization_6/mul_1:z:0.layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_6/Reshape/shape?
layer_normalization_6/ReshapeReshapeinputs,layer_normalization_6/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_6/Reshape?
!layer_normalization_6/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!layer_normalization_6/ones/Less/y?
layer_normalization_6/ones/LessLesslayer_normalization_6/mul:z:0*layer_normalization_6/ones/Less/y:output:0*
T0*
_output_shapes
: 2!
layer_normalization_6/ones/Less?
!layer_normalization_6/ones/packedPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_6/ones/packed?
 layer_normalization_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 layer_normalization_6/ones/Const?
layer_normalization_6/onesFill*layer_normalization_6/ones/packed:output:0)layer_normalization_6/ones/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_6/ones?
"layer_normalization_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"layer_normalization_6/zeros/Less/y?
 layer_normalization_6/zeros/LessLesslayer_normalization_6/mul:z:0+layer_normalization_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 layer_normalization_6/zeros/Less?
"layer_normalization_6/zeros/packedPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2$
"layer_normalization_6/zeros/packed?
!layer_normalization_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!layer_normalization_6/zeros/Const?
layer_normalization_6/zerosFill+layer_normalization_6/zeros/packed:output:0*layer_normalization_6/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_6/zeros}
layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_6/Const?
layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_6/Const_1?
&layer_normalization_6/FusedBatchNormV3FusedBatchNormV3&layer_normalization_6/Reshape:output:0#layer_normalization_6/ones:output:0$layer_normalization_6/zeros:output:0$layer_normalization_6/Const:output:0&layer_normalization_6/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_6/FusedBatchNormV3?
layer_normalization_6/Reshape_1Reshape*layer_normalization_6/FusedBatchNormV3:y:0$layer_normalization_6/Shape:output:0*
T0*'
_output_shapes
:?????????2!
layer_normalization_6/Reshape_1?
*layer_normalization_6/mul_2/ReadVariableOpReadVariableOp3layer_normalization_6_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02,
*layer_normalization_6/mul_2/ReadVariableOp?
layer_normalization_6/mul_2Mul(layer_normalization_6/Reshape_1:output:02layer_normalization_6/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_6/mul_2?
(layer_normalization_6/add/ReadVariableOpReadVariableOp1layer_normalization_6_add_readvariableop_resource*
_output_shapes
:*
dtype02*
(layer_normalization_6/add/ReadVariableOp?
layer_normalization_6/addAddV2layer_normalization_6/mul_2:z:00layer_normalization_6/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_6/add?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMullayer_normalization_6/add:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_6/Relu?
layer_normalization_7/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
layer_normalization_7/Shape?
)layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_7/strided_slice/stack?
+layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice/stack_1?
+layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice/stack_2?
#layer_normalization_7/strided_sliceStridedSlice$layer_normalization_7/Shape:output:02layer_normalization_7/strided_slice/stack:output:04layer_normalization_7/strided_slice/stack_1:output:04layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_7/strided_slice|
layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_7/mul/x?
layer_normalization_7/mulMul$layer_normalization_7/mul/x:output:0,layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_7/mul?
+layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice_1/stack?
-layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_7/strided_slice_1/stack_1?
-layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_7/strided_slice_1/stack_2?
%layer_normalization_7/strided_slice_1StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_1/stack:output:06layer_normalization_7/strided_slice_1/stack_1:output:06layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_7/strided_slice_1?
layer_normalization_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_7/mul_1/x?
layer_normalization_7/mul_1Mul&layer_normalization_7/mul_1/x:output:0.layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_7/mul_1?
%layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_7/Reshape/shape/0?
%layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_7/Reshape/shape/3?
#layer_normalization_7/Reshape/shapePack.layer_normalization_7/Reshape/shape/0:output:0layer_normalization_7/mul:z:0layer_normalization_7/mul_1:z:0.layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_7/Reshape/shape?
layer_normalization_7/ReshapeReshapedense_6/Relu:activations:0,layer_normalization_7/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_7/Reshape?
!layer_normalization_7/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!layer_normalization_7/ones/Less/y?
layer_normalization_7/ones/LessLesslayer_normalization_7/mul:z:0*layer_normalization_7/ones/Less/y:output:0*
T0*
_output_shapes
: 2!
layer_normalization_7/ones/Less?
!layer_normalization_7/ones/packedPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_7/ones/packed?
 layer_normalization_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 layer_normalization_7/ones/Const?
layer_normalization_7/onesFill*layer_normalization_7/ones/packed:output:0)layer_normalization_7/ones/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_7/ones?
"layer_normalization_7/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"layer_normalization_7/zeros/Less/y?
 layer_normalization_7/zeros/LessLesslayer_normalization_7/mul:z:0+layer_normalization_7/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 layer_normalization_7/zeros/Less?
"layer_normalization_7/zeros/packedPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2$
"layer_normalization_7/zeros/packed?
!layer_normalization_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!layer_normalization_7/zeros/Const?
layer_normalization_7/zerosFill+layer_normalization_7/zeros/packed:output:0*layer_normalization_7/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_7/zeros}
layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_7/Const?
layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_7/Const_1?
&layer_normalization_7/FusedBatchNormV3FusedBatchNormV3&layer_normalization_7/Reshape:output:0#layer_normalization_7/ones:output:0$layer_normalization_7/zeros:output:0$layer_normalization_7/Const:output:0&layer_normalization_7/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_7/FusedBatchNormV3?
layer_normalization_7/Reshape_1Reshape*layer_normalization_7/FusedBatchNormV3:y:0$layer_normalization_7/Shape:output:0*
T0*'
_output_shapes
:?????????@2!
layer_normalization_7/Reshape_1?
*layer_normalization_7/mul_2/ReadVariableOpReadVariableOp3layer_normalization_7_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype02,
*layer_normalization_7/mul_2/ReadVariableOp?
layer_normalization_7/mul_2Mul(layer_normalization_7/Reshape_1:output:02layer_normalization_7/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization_7/mul_2?
(layer_normalization_7/add/ReadVariableOpReadVariableOp1layer_normalization_7_add_readvariableop_resource*
_output_shapes
:@*
dtype02*
(layer_normalization_7/add/ReadVariableOp?
layer_normalization_7/addAddV2layer_normalization_7/mul_2:z:00layer_normalization_7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization_7/add?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMullayer_normalization_7/add:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_7/Relu?
layer_normalization_8/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
layer_normalization_8/Shape?
)layer_normalization_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_8/strided_slice/stack?
+layer_normalization_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_8/strided_slice/stack_1?
+layer_normalization_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_8/strided_slice/stack_2?
#layer_normalization_8/strided_sliceStridedSlice$layer_normalization_8/Shape:output:02layer_normalization_8/strided_slice/stack:output:04layer_normalization_8/strided_slice/stack_1:output:04layer_normalization_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_8/strided_slice|
layer_normalization_8/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_8/mul/x?
layer_normalization_8/mulMul$layer_normalization_8/mul/x:output:0,layer_normalization_8/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_8/mul?
+layer_normalization_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_8/strided_slice_1/stack?
-layer_normalization_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_8/strided_slice_1/stack_1?
-layer_normalization_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_8/strided_slice_1/stack_2?
%layer_normalization_8/strided_slice_1StridedSlice$layer_normalization_8/Shape:output:04layer_normalization_8/strided_slice_1/stack:output:06layer_normalization_8/strided_slice_1/stack_1:output:06layer_normalization_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_8/strided_slice_1?
layer_normalization_8/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_8/mul_1/x?
layer_normalization_8/mul_1Mul&layer_normalization_8/mul_1/x:output:0.layer_normalization_8/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_8/mul_1?
%layer_normalization_8/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_8/Reshape/shape/0?
%layer_normalization_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_8/Reshape/shape/3?
#layer_normalization_8/Reshape/shapePack.layer_normalization_8/Reshape/shape/0:output:0layer_normalization_8/mul:z:0layer_normalization_8/mul_1:z:0.layer_normalization_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_8/Reshape/shape?
layer_normalization_8/ReshapeReshapedense_7/Relu:activations:0,layer_normalization_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_8/Reshape?
!layer_normalization_8/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!layer_normalization_8/ones/Less/y?
layer_normalization_8/ones/LessLesslayer_normalization_8/mul:z:0*layer_normalization_8/ones/Less/y:output:0*
T0*
_output_shapes
: 2!
layer_normalization_8/ones/Less?
!layer_normalization_8/ones/packedPacklayer_normalization_8/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_8/ones/packed?
 layer_normalization_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 layer_normalization_8/ones/Const?
layer_normalization_8/onesFill*layer_normalization_8/ones/packed:output:0)layer_normalization_8/ones/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_8/ones?
"layer_normalization_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"layer_normalization_8/zeros/Less/y?
 layer_normalization_8/zeros/LessLesslayer_normalization_8/mul:z:0+layer_normalization_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 layer_normalization_8/zeros/Less?
"layer_normalization_8/zeros/packedPacklayer_normalization_8/mul:z:0*
N*
T0*
_output_shapes
:2$
"layer_normalization_8/zeros/packed?
!layer_normalization_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!layer_normalization_8/zeros/Const?
layer_normalization_8/zerosFill+layer_normalization_8/zeros/packed:output:0*layer_normalization_8/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_8/zeros}
layer_normalization_8/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_8/Const?
layer_normalization_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_8/Const_1?
&layer_normalization_8/FusedBatchNormV3FusedBatchNormV3&layer_normalization_8/Reshape:output:0#layer_normalization_8/ones:output:0$layer_normalization_8/zeros:output:0$layer_normalization_8/Const:output:0&layer_normalization_8/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_8/FusedBatchNormV3?
layer_normalization_8/Reshape_1Reshape*layer_normalization_8/FusedBatchNormV3:y:0$layer_normalization_8/Shape:output:0*
T0*'
_output_shapes
:?????????@2!
layer_normalization_8/Reshape_1?
*layer_normalization_8/mul_2/ReadVariableOpReadVariableOp3layer_normalization_8_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype02,
*layer_normalization_8/mul_2/ReadVariableOp?
layer_normalization_8/mul_2Mul(layer_normalization_8/Reshape_1:output:02layer_normalization_8/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization_8/mul_2?
(layer_normalization_8/add/ReadVariableOpReadVariableOp1layer_normalization_8_add_readvariableop_resource*
_output_shapes
:@*
dtype02*
(layer_normalization_8/add/ReadVariableOp?
layer_normalization_8/addAddV2layer_normalization_8/mul_2:z:00layer_normalization_8/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization_8/add?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMullayer_normalization_8/add:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAddp
dense_8/TanhTanhdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_8/Tanhm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x?
rescaling_1/mulMuldense_8/Tanh:y:0rescaling_1/Cast/x:output:0*
T0*'
_output_shapes
:?????????2
rescaling_1/mul?
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*'
_output_shapes
:?????????2
rescaling_1/addn
IdentityIdentityrescaling_1/add:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp)^layer_normalization_6/add/ReadVariableOp+^layer_normalization_6/mul_2/ReadVariableOp)^layer_normalization_7/add/ReadVariableOp+^layer_normalization_7/mul_2/ReadVariableOp)^layer_normalization_8/add/ReadVariableOp+^layer_normalization_8/mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2T
(layer_normalization_6/add/ReadVariableOp(layer_normalization_6/add/ReadVariableOp2X
*layer_normalization_6/mul_2/ReadVariableOp*layer_normalization_6/mul_2/ReadVariableOp2T
(layer_normalization_7/add/ReadVariableOp(layer_normalization_7/add/ReadVariableOp2X
*layer_normalization_7/mul_2/ReadVariableOp*layer_normalization_7/mul_2/ReadVariableOp2T
(layer_normalization_8/add/ReadVariableOp(layer_normalization_8/add/ReadVariableOp2X
*layer_normalization_8/mul_2/ReadVariableOp*layer_normalization_8/mul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_8_layer_call_fn_4102797

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_41018822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_model_2_layer_call_fn_4102244

inputs
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_41018992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_model_2_layer_call_fn_4101926
input_3
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_41018992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
??
?

D__inference_model_2_layer_call_and_return_conditional_losses_4102428

inputsA
3layer_normalization_6_mul_2_readvariableop_resource:?
1layer_normalization_6_add_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:@5
'dense_6_biasadd_readvariableop_resource:@A
3layer_normalization_7_mul_2_readvariableop_resource:@?
1layer_normalization_7_add_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@A
3layer_normalization_8_mul_2_readvariableop_resource:@?
1layer_normalization_8_add_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@5
'dense_8_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?(layer_normalization_6/add/ReadVariableOp?*layer_normalization_6/mul_2/ReadVariableOp?(layer_normalization_7/add/ReadVariableOp?*layer_normalization_7/mul_2/ReadVariableOp?(layer_normalization_8/add/ReadVariableOp?*layer_normalization_8/mul_2/ReadVariableOpp
layer_normalization_6/ShapeShapeinputs*
T0*
_output_shapes
:2
layer_normalization_6/Shape?
)layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_6/strided_slice/stack?
+layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice/stack_1?
+layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice/stack_2?
#layer_normalization_6/strided_sliceStridedSlice$layer_normalization_6/Shape:output:02layer_normalization_6/strided_slice/stack:output:04layer_normalization_6/strided_slice/stack_1:output:04layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_6/strided_slice|
layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_6/mul/x?
layer_normalization_6/mulMul$layer_normalization_6/mul/x:output:0,layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_6/mul?
+layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_6/strided_slice_1/stack?
-layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_6/strided_slice_1/stack_1?
-layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_6/strided_slice_1/stack_2?
%layer_normalization_6/strided_slice_1StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_1/stack:output:06layer_normalization_6/strided_slice_1/stack_1:output:06layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_6/strided_slice_1?
layer_normalization_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_6/mul_1/x?
layer_normalization_6/mul_1Mul&layer_normalization_6/mul_1/x:output:0.layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_6/mul_1?
%layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_6/Reshape/shape/0?
%layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_6/Reshape/shape/3?
#layer_normalization_6/Reshape/shapePack.layer_normalization_6/Reshape/shape/0:output:0layer_normalization_6/mul:z:0layer_normalization_6/mul_1:z:0.layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_6/Reshape/shape?
layer_normalization_6/ReshapeReshapeinputs,layer_normalization_6/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_6/Reshape?
!layer_normalization_6/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!layer_normalization_6/ones/Less/y?
layer_normalization_6/ones/LessLesslayer_normalization_6/mul:z:0*layer_normalization_6/ones/Less/y:output:0*
T0*
_output_shapes
: 2!
layer_normalization_6/ones/Less?
!layer_normalization_6/ones/packedPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_6/ones/packed?
 layer_normalization_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 layer_normalization_6/ones/Const?
layer_normalization_6/onesFill*layer_normalization_6/ones/packed:output:0)layer_normalization_6/ones/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_6/ones?
"layer_normalization_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"layer_normalization_6/zeros/Less/y?
 layer_normalization_6/zeros/LessLesslayer_normalization_6/mul:z:0+layer_normalization_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 layer_normalization_6/zeros/Less?
"layer_normalization_6/zeros/packedPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:2$
"layer_normalization_6/zeros/packed?
!layer_normalization_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!layer_normalization_6/zeros/Const?
layer_normalization_6/zerosFill+layer_normalization_6/zeros/packed:output:0*layer_normalization_6/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_6/zeros}
layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_6/Const?
layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_6/Const_1?
&layer_normalization_6/FusedBatchNormV3FusedBatchNormV3&layer_normalization_6/Reshape:output:0#layer_normalization_6/ones:output:0$layer_normalization_6/zeros:output:0$layer_normalization_6/Const:output:0&layer_normalization_6/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_6/FusedBatchNormV3?
layer_normalization_6/Reshape_1Reshape*layer_normalization_6/FusedBatchNormV3:y:0$layer_normalization_6/Shape:output:0*
T0*'
_output_shapes
:?????????2!
layer_normalization_6/Reshape_1?
*layer_normalization_6/mul_2/ReadVariableOpReadVariableOp3layer_normalization_6_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02,
*layer_normalization_6/mul_2/ReadVariableOp?
layer_normalization_6/mul_2Mul(layer_normalization_6/Reshape_1:output:02layer_normalization_6/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_6/mul_2?
(layer_normalization_6/add/ReadVariableOpReadVariableOp1layer_normalization_6_add_readvariableop_resource*
_output_shapes
:*
dtype02*
(layer_normalization_6/add/ReadVariableOp?
layer_normalization_6/addAddV2layer_normalization_6/mul_2:z:00layer_normalization_6/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
layer_normalization_6/add?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMullayer_normalization_6/add:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_6/Relu?
layer_normalization_7/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
layer_normalization_7/Shape?
)layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_7/strided_slice/stack?
+layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice/stack_1?
+layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice/stack_2?
#layer_normalization_7/strided_sliceStridedSlice$layer_normalization_7/Shape:output:02layer_normalization_7/strided_slice/stack:output:04layer_normalization_7/strided_slice/stack_1:output:04layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_7/strided_slice|
layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_7/mul/x?
layer_normalization_7/mulMul$layer_normalization_7/mul/x:output:0,layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_7/mul?
+layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_7/strided_slice_1/stack?
-layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_7/strided_slice_1/stack_1?
-layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_7/strided_slice_1/stack_2?
%layer_normalization_7/strided_slice_1StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_1/stack:output:06layer_normalization_7/strided_slice_1/stack_1:output:06layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_7/strided_slice_1?
layer_normalization_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_7/mul_1/x?
layer_normalization_7/mul_1Mul&layer_normalization_7/mul_1/x:output:0.layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_7/mul_1?
%layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_7/Reshape/shape/0?
%layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_7/Reshape/shape/3?
#layer_normalization_7/Reshape/shapePack.layer_normalization_7/Reshape/shape/0:output:0layer_normalization_7/mul:z:0layer_normalization_7/mul_1:z:0.layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_7/Reshape/shape?
layer_normalization_7/ReshapeReshapedense_6/Relu:activations:0,layer_normalization_7/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_7/Reshape?
!layer_normalization_7/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!layer_normalization_7/ones/Less/y?
layer_normalization_7/ones/LessLesslayer_normalization_7/mul:z:0*layer_normalization_7/ones/Less/y:output:0*
T0*
_output_shapes
: 2!
layer_normalization_7/ones/Less?
!layer_normalization_7/ones/packedPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_7/ones/packed?
 layer_normalization_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 layer_normalization_7/ones/Const?
layer_normalization_7/onesFill*layer_normalization_7/ones/packed:output:0)layer_normalization_7/ones/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_7/ones?
"layer_normalization_7/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"layer_normalization_7/zeros/Less/y?
 layer_normalization_7/zeros/LessLesslayer_normalization_7/mul:z:0+layer_normalization_7/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 layer_normalization_7/zeros/Less?
"layer_normalization_7/zeros/packedPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:2$
"layer_normalization_7/zeros/packed?
!layer_normalization_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!layer_normalization_7/zeros/Const?
layer_normalization_7/zerosFill+layer_normalization_7/zeros/packed:output:0*layer_normalization_7/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_7/zeros}
layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_7/Const?
layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_7/Const_1?
&layer_normalization_7/FusedBatchNormV3FusedBatchNormV3&layer_normalization_7/Reshape:output:0#layer_normalization_7/ones:output:0$layer_normalization_7/zeros:output:0$layer_normalization_7/Const:output:0&layer_normalization_7/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_7/FusedBatchNormV3?
layer_normalization_7/Reshape_1Reshape*layer_normalization_7/FusedBatchNormV3:y:0$layer_normalization_7/Shape:output:0*
T0*'
_output_shapes
:?????????@2!
layer_normalization_7/Reshape_1?
*layer_normalization_7/mul_2/ReadVariableOpReadVariableOp3layer_normalization_7_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype02,
*layer_normalization_7/mul_2/ReadVariableOp?
layer_normalization_7/mul_2Mul(layer_normalization_7/Reshape_1:output:02layer_normalization_7/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization_7/mul_2?
(layer_normalization_7/add/ReadVariableOpReadVariableOp1layer_normalization_7_add_readvariableop_resource*
_output_shapes
:@*
dtype02*
(layer_normalization_7/add/ReadVariableOp?
layer_normalization_7/addAddV2layer_normalization_7/mul_2:z:00layer_normalization_7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization_7/add?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMullayer_normalization_7/add:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_7/Relu?
layer_normalization_8/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:2
layer_normalization_8/Shape?
)layer_normalization_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_8/strided_slice/stack?
+layer_normalization_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_8/strided_slice/stack_1?
+layer_normalization_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_8/strided_slice/stack_2?
#layer_normalization_8/strided_sliceStridedSlice$layer_normalization_8/Shape:output:02layer_normalization_8/strided_slice/stack:output:04layer_normalization_8/strided_slice/stack_1:output:04layer_normalization_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_8/strided_slice|
layer_normalization_8/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_8/mul/x?
layer_normalization_8/mulMul$layer_normalization_8/mul/x:output:0,layer_normalization_8/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_8/mul?
+layer_normalization_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_8/strided_slice_1/stack?
-layer_normalization_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_8/strided_slice_1/stack_1?
-layer_normalization_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_8/strided_slice_1/stack_2?
%layer_normalization_8/strided_slice_1StridedSlice$layer_normalization_8/Shape:output:04layer_normalization_8/strided_slice_1/stack:output:06layer_normalization_8/strided_slice_1/stack_1:output:06layer_normalization_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_8/strided_slice_1?
layer_normalization_8/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_8/mul_1/x?
layer_normalization_8/mul_1Mul&layer_normalization_8/mul_1/x:output:0.layer_normalization_8/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_8/mul_1?
%layer_normalization_8/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_8/Reshape/shape/0?
%layer_normalization_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_8/Reshape/shape/3?
#layer_normalization_8/Reshape/shapePack.layer_normalization_8/Reshape/shape/0:output:0layer_normalization_8/mul:z:0layer_normalization_8/mul_1:z:0.layer_normalization_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_8/Reshape/shape?
layer_normalization_8/ReshapeReshapedense_7/Relu:activations:0,layer_normalization_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization_8/Reshape?
!layer_normalization_8/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!layer_normalization_8/ones/Less/y?
layer_normalization_8/ones/LessLesslayer_normalization_8/mul:z:0*layer_normalization_8/ones/Less/y:output:0*
T0*
_output_shapes
: 2!
layer_normalization_8/ones/Less?
!layer_normalization_8/ones/packedPacklayer_normalization_8/mul:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_8/ones/packed?
 layer_normalization_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 layer_normalization_8/ones/Const?
layer_normalization_8/onesFill*layer_normalization_8/ones/packed:output:0)layer_normalization_8/ones/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_8/ones?
"layer_normalization_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"layer_normalization_8/zeros/Less/y?
 layer_normalization_8/zeros/LessLesslayer_normalization_8/mul:z:0+layer_normalization_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 layer_normalization_8/zeros/Less?
"layer_normalization_8/zeros/packedPacklayer_normalization_8/mul:z:0*
N*
T0*
_output_shapes
:2$
"layer_normalization_8/zeros/packed?
!layer_normalization_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!layer_normalization_8/zeros/Const?
layer_normalization_8/zerosFill+layer_normalization_8/zeros/packed:output:0*layer_normalization_8/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization_8/zeros}
layer_normalization_8/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_8/Const?
layer_normalization_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_8/Const_1?
&layer_normalization_8/FusedBatchNormV3FusedBatchNormV3&layer_normalization_8/Reshape:output:0#layer_normalization_8/ones:output:0$layer_normalization_8/zeros:output:0$layer_normalization_8/Const:output:0&layer_normalization_8/Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2(
&layer_normalization_8/FusedBatchNormV3?
layer_normalization_8/Reshape_1Reshape*layer_normalization_8/FusedBatchNormV3:y:0$layer_normalization_8/Shape:output:0*
T0*'
_output_shapes
:?????????@2!
layer_normalization_8/Reshape_1?
*layer_normalization_8/mul_2/ReadVariableOpReadVariableOp3layer_normalization_8_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype02,
*layer_normalization_8/mul_2/ReadVariableOp?
layer_normalization_8/mul_2Mul(layer_normalization_8/Reshape_1:output:02layer_normalization_8/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization_8/mul_2?
(layer_normalization_8/add/ReadVariableOpReadVariableOp1layer_normalization_8_add_readvariableop_resource*
_output_shapes
:@*
dtype02*
(layer_normalization_8/add/ReadVariableOp?
layer_normalization_8/addAddV2layer_normalization_8/mul_2:z:00layer_normalization_8/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization_8/add?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMullayer_normalization_8/add:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAddp
dense_8/TanhTanhdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_8/Tanhm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x?
rescaling_1/mulMuldense_8/Tanh:y:0rescaling_1/Cast/x:output:0*
T0*'
_output_shapes
:?????????2
rescaling_1/mul?
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*'
_output_shapes
:?????????2
rescaling_1/addn
IdentityIdentityrescaling_1/add:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp)^layer_normalization_6/add/ReadVariableOp+^layer_normalization_6/mul_2/ReadVariableOp)^layer_normalization_7/add/ReadVariableOp+^layer_normalization_7/mul_2/ReadVariableOp)^layer_normalization_8/add/ReadVariableOp+^layer_normalization_8/mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2T
(layer_normalization_6/add/ReadVariableOp(layer_normalization_6/add/ReadVariableOp2X
*layer_normalization_6/mul_2/ReadVariableOp*layer_normalization_6/mul_2/ReadVariableOp2T
(layer_normalization_7/add/ReadVariableOp(layer_normalization_7/add/ReadVariableOp2X
*layer_normalization_7/mul_2/ReadVariableOp*layer_normalization_7/mul_2/ReadVariableOp2T
(layer_normalization_8/add/ReadVariableOp(layer_normalization_8/add/ReadVariableOp2X
*layer_normalization_8/mul_2/ReadVariableOp*layer_normalization_8/mul_2/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_layer_normalization_7_layer_call_fn_4102667

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_layer_normalization_7_layer_call_and_return_conditional_losses_41017962
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
%__inference_signature_wrapper_4102215
input_3
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_41016742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?'
?
R__inference_layer_normalization_8_layer_call_and_return_conditional_losses_4102788

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
Reshape]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/y^
	ones/LessLessmul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/LessY
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Constm
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????2
ones_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/ya

zeros/LessLessmul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Less[
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constq
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????2
zerosQ
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@2
	Reshape_1?
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype02
mul_2/ReadVariableOpy
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
mul_2?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype02
add/ReadVariableOpl
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityz
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_dense_7_layer_call_and_return_conditional_losses_4101813

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_dense_6_layer_call_and_return_conditional_losses_4101744

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_30
serving_default_input_3:0??????????
rescaling_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?{
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api

signatures
a__call__
*b&call_and_return_all_conditional_losses
c_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?
axis
	gamma
beta
	variables
trainable_variables
regularization_losses
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	gamma
beta
	variables
trainable_variables
 regularization_losses
!	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
?

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(axis
	)gamma
*beta
+	variables
,trainable_variables
-regularization_losses
.	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
?

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
v
0
1
2
3
4
5
"6
#7
)8
*9
/10
011"
trackable_list_wrapper
v
0
1
2
3
4
5
"6
#7
)8
*9
/10
011"
trackable_list_wrapper
 "
trackable_list_wrapper
?
		variables
9layer_regularization_losses
:metrics
;layer_metrics

trainable_variables
<non_trainable_variables

=layers
regularization_losses
a__call__
c_default_save_signature
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
,
rserving_default"
signature_map
 "
trackable_list_wrapper
):'2layer_normalization_6/gamma
(:&2layer_normalization_6/beta
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
>layer_regularization_losses
?metrics
@layer_metrics
trainable_variables
Anon_trainable_variables

Blayers
regularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_6/kernel
:@2dense_6/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Clayer_regularization_losses
Dmetrics
Elayer_metrics
trainable_variables
Fnon_trainable_variables

Glayers
regularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_7/gamma
(:&@2layer_normalization_7/beta
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Hlayer_regularization_losses
Imetrics
Jlayer_metrics
trainable_variables
Knon_trainable_variables

Llayers
 regularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 :@@2dense_7/kernel
:@2dense_7/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
$	variables
Mlayer_regularization_losses
Nmetrics
Olayer_metrics
%trainable_variables
Pnon_trainable_variables

Qlayers
&regularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_8/gamma
(:&@2layer_normalization_8/beta
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+	variables
Rlayer_regularization_losses
Smetrics
Tlayer_metrics
,trainable_variables
Unon_trainable_variables

Vlayers
-regularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_8/kernel
:2dense_8/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
1	variables
Wlayer_regularization_losses
Xmetrics
Ylayer_metrics
2trainable_variables
Znon_trainable_variables

[layers
3regularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
5	variables
\layer_regularization_losses
]metrics
^layer_metrics
6trainable_variables
_non_trainable_variables

`layers
7regularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
)__inference_model_2_layer_call_fn_4101926
)__inference_model_2_layer_call_fn_4102244
)__inference_model_2_layer_call_fn_4102273
)__inference_model_2_layer_call_fn_4102114?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_model_2_layer_call_and_return_conditional_losses_4102428
D__inference_model_2_layer_call_and_return_conditional_losses_4102583
D__inference_model_2_layer_call_and_return_conditional_losses_4102149
D__inference_model_2_layer_call_and_return_conditional_losses_4102184?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_4101674input_3"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_layer_normalization_6_layer_call_fn_4102592?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_layer_normalization_6_layer_call_and_return_conditional_losses_4102638?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_6_layer_call_fn_4102647?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_6_layer_call_and_return_conditional_losses_4102658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_layer_normalization_7_layer_call_fn_4102667?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_layer_normalization_7_layer_call_and_return_conditional_losses_4102713?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_7_layer_call_fn_4102722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_7_layer_call_and_return_conditional_losses_4102733?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_layer_normalization_8_layer_call_fn_4102742?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_layer_normalization_8_layer_call_and_return_conditional_losses_4102788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_8_layer_call_fn_4102797?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_8_layer_call_and_return_conditional_losses_4102808?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_rescaling_1_layer_call_fn_4102813?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_rescaling_1_layer_call_and_return_conditional_losses_4102821?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_4102215input_3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_4101674{"#)*/00?-
&?#
!?
input_3?????????
? "9?6
4
rescaling_1%?"
rescaling_1??????????
D__inference_dense_6_layer_call_and_return_conditional_losses_4102658\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? |
)__inference_dense_6_layer_call_fn_4102647O/?,
%?"
 ?
inputs?????????
? "??????????@?
D__inference_dense_7_layer_call_and_return_conditional_losses_4102733\"#/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? |
)__inference_dense_7_layer_call_fn_4102722O"#/?,
%?"
 ?
inputs?????????@
? "??????????@?
D__inference_dense_8_layer_call_and_return_conditional_losses_4102808\/0/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? |
)__inference_dense_8_layer_call_fn_4102797O/0/?,
%?"
 ?
inputs?????????@
? "???????????
R__inference_layer_normalization_6_layer_call_and_return_conditional_losses_4102638\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
7__inference_layer_normalization_6_layer_call_fn_4102592O/?,
%?"
 ?
inputs?????????
? "???????????
R__inference_layer_normalization_7_layer_call_and_return_conditional_losses_4102713\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
7__inference_layer_normalization_7_layer_call_fn_4102667O/?,
%?"
 ?
inputs?????????@
? "??????????@?
R__inference_layer_normalization_8_layer_call_and_return_conditional_losses_4102788\)*/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
7__inference_layer_normalization_8_layer_call_fn_4102742O)*/?,
%?"
 ?
inputs?????????@
? "??????????@?
D__inference_model_2_layer_call_and_return_conditional_losses_4102149o"#)*/08?5
.?+
!?
input_3?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_2_layer_call_and_return_conditional_losses_4102184o"#)*/08?5
.?+
!?
input_3?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_model_2_layer_call_and_return_conditional_losses_4102428n"#)*/07?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_2_layer_call_and_return_conditional_losses_4102583n"#)*/07?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
)__inference_model_2_layer_call_fn_4101926b"#)*/08?5
.?+
!?
input_3?????????
p 

 
? "???????????
)__inference_model_2_layer_call_fn_4102114b"#)*/08?5
.?+
!?
input_3?????????
p

 
? "???????????
)__inference_model_2_layer_call_fn_4102244a"#)*/07?4
-?*
 ?
inputs?????????
p 

 
? "???????????
)__inference_model_2_layer_call_fn_4102273a"#)*/07?4
-?*
 ?
inputs?????????
p

 
? "???????????
H__inference_rescaling_1_layer_call_and_return_conditional_losses_4102821X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_rescaling_1_layer_call_fn_4102813K/?,
%?"
 ?
inputs?????????
? "???????????
%__inference_signature_wrapper_4102215?"#)*/0;?8
? 
1?.
,
input_3!?
input_3?????????"9?6
4
rescaling_1%?"
rescaling_1?????????