pub mod activation;
pub mod advanced;
pub mod arithmetic;
pub mod audio;
pub mod comparison;
pub mod conv;
pub mod linear;
pub mod normalization;
pub mod reduction;
pub mod rnn;
pub mod sequence;
pub mod shape;
pub mod unary;

/// Trait for ONNX operator execution
pub trait Operator<B: burn::prelude::Backend>: Send + Sync {
    fn name(&self) -> &'static str;
    fn execute(
        &self,
        inputs: Vec<String>,
        outputs: Vec<String>,
        config: OperatorConfig,
        values: &mut crate::runtime::value_store::ValueStore<B>,
        device: &B::Device,
    ) -> anyhow::Result<()>;
}

#[derive(Clone)]
pub enum OperatorConfig {
    // Placeholder - will have specific configs for each operator type
    Conv,
    Pool,
    Linear,
    Activation,
    Shape,
}

/// Dispatcher for ONNX operators
pub fn dispatch<B: burn::prelude::Backend>(
    node: &onnx_ir::ir::Node,
    values: &mut crate::runtime::value_store::ValueStore<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    use onnx_ir::ir::Node;

    // Skip Constant nodes - they're just weight storage
    if let Node::Constant(_) = node {
        return Ok(());
    }

    match node {
        // Unary math operations
        Node::Abs(_) => {
            crate::ops::unary::abs(node, values, device)?;
        }
        Node::Neg(_) => {
            crate::ops::unary::neg(node, values, device)?;
        }
        Node::Sqrt(_) => {
            crate::ops::unary::sqrt(node, values, device)?;
        }
        Node::Exp(_) => {
            crate::ops::unary::exp(node, values, device)?;
        }
        Node::Log(_) => {
            crate::ops::unary::log(node, values, device)?;
        }
        Node::Ceil(_) => {
            crate::ops::unary::ceil(node, values, device)?;
        }
        Node::Floor(_) => {
            crate::ops::unary::floor(node, values, device)?;
        }
        Node::Round(_) => {
            crate::ops::unary::round(node, values, device)?;
        }
        Node::Sign(_) => {
            crate::ops::unary::sign(node, values, device)?;
        }
        Node::Reciprocal(_) => {
            crate::ops::unary::reciprocal(node, values, device)?;
        }
        Node::Sin(_) => {
            crate::ops::unary::sin(node, values, device)?;
        }
        Node::Cos(_) => {
            crate::ops::unary::cos(node, values, device)?;
        }
        Node::Tan(_) => {
            crate::ops::unary::tan(node, values, device)?;
        }
        Node::Sinh(_) => {
            crate::ops::unary::sinh(node, values, device)?;
        }
        Node::Cosh(_) => {
            crate::ops::unary::cosh(node, values, device)?;
        }
        Node::Erf(_) => {
            crate::ops::unary::erf(node, values, device)?;
        }
        Node::Atan(_) => {
            crate::ops::unary::atan(node, values, device)?;
        }

        // Phase 1 operators
        Node::Add(_) => {
            crate::ops::arithmetic::add(node, values, device)?;
        }
        Node::Sub(_) => {
            crate::ops::arithmetic::sub(node, values, device)?;
        }
        Node::Mul(_) => {
            crate::ops::arithmetic::mul(node, values, device)?;
        }
        Node::Div(_) => {
            crate::ops::arithmetic::div(node, values, device)?;
        }
        Node::Relu(_) => {
            crate::ops::activation::relu(node, values, device)?;
        }
        Node::Sigmoid(_) => {
            crate::ops::activation::sigmoid(node, values, device)?;
        }
        Node::Softmax(_) => {
            crate::ops::activation::softmax(node, values, device)?;
        }
        Node::Tanh(_) => {
            crate::ops::activation::tanh(node, values, device)?;
        }
        Node::Gelu(_) => {
            crate::ops::activation::gelu(node, values, device)?;
        }
        Node::LeakyRelu(_) => {
            crate::ops::activation::leaky_relu(node, values, device)?;
        }
        Node::HardSigmoid(_) => {
            crate::ops::activation::hard_sigmoid(node, values, device)?;
        }
        Node::HardSwish(_) => {
            crate::ops::activation::hard_swish(node, values, device)?;
        }
        Node::PRelu(_) => {
            crate::ops::activation::prelu(node, values, device)?;
        }
        Node::LogSoftmax(_) => {
            crate::ops::activation::log_softmax(node, values, device)?;
        }
        Node::Identity(_) => {
            crate::ops::activation::identity_(node, values, device)?;
        }
        Node::Dropout(_) => {
            crate::ops::activation::dropout(node, values, device)?;
        }
        Node::Clip(_) => {
            crate::ops::activation::clip(node, values, device)?;
        }
        Node::Conv2d(_) => {
            crate::ops::conv::conv2d(node, values, device)?;
        }
        Node::Conv1d(_) => {
            crate::ops::conv::conv1d(node, values, device)?;
        }
        Node::ConvTranspose2d(_) => {
            crate::ops::conv::conv_transpose2d(node, values, device)?;
        }
        Node::ConvTranspose1d(_) => {
            crate::ops::conv::conv_transpose1d(node, values, device)?;
        }
        Node::ConvTranspose(_) => {
            crate::ops::conv::conv_transpose(node, values, device)?;
        }
        Node::MaxPool2d(_) => {
            crate::ops::conv::max_pool_2d(node, values, device)?;
        }
        Node::MaxPool1d(_) => {
            crate::ops::conv::max_pool_1d(node, values, device)?;
        }
        Node::AveragePool2d(_) => {
            crate::ops::conv::avg_pool_2d(node, values, device)?;
        }
        Node::AveragePool1d(_) => {
            crate::ops::conv::avg_pool_1d(node, values, device)?;
        }
        Node::MatMul(_) => {
            crate::ops::linear::matmul(node, values, device)?;
        }
        Node::Gemm(_) => {
            crate::ops::linear::gemm(node, values, device)?;
        }
        Node::Linear(_) => {
            crate::ops::linear::linear(node, values, device)?;
        }
        Node::Flatten(_) => {
            crate::ops::shape::flatten(node, values, device)?;
        }
        Node::Reshape(_) => {
            crate::ops::shape::reshape(node, values, device)?;
        }
        Node::Shape(_) => {
            crate::ops::shape::shape(node, values, device)?;
        }
        // Phase 2 operators
        Node::BatchNormalization(_) => {
            crate::ops::conv::batch_normalization(node, values, device)?;
        }
        // Normalization operators
        Node::LayerNormalization(_) => {
            crate::ops::normalization::layer_norm(node, values, device)?;
        }
        Node::InstanceNormalization(_) => {
            crate::ops::normalization::instance_norm(node, values, device)?;
        }
        Node::GroupNormalization(_) => {
            crate::ops::normalization::group_norm(node, values, device)?;
        }
        Node::GlobalAveragePool(_) => {
            crate::ops::conv::global_average_pool(node, values, device)?;
        }
        Node::Concat(_) => {
            crate::ops::shape::concat(node, values, device)?;
        }
        Node::Transpose(_) => {
            crate::ops::shape::transpose(node, values, device)?;
        }
        Node::Squeeze(_) => {
            crate::ops::shape::squeeze(node, values, device)?;
        }
        Node::Unsqueeze(_) => {
            crate::ops::shape::unsqueeze(node, values, device)?;
        }
        Node::Size(_) => {
            crate::ops::shape::size(node, values, device)?;
        }
        // Reduction operators
        Node::ReduceSum(_) => {
            crate::ops::reduction::reduce_sum(node, values, device)?;
        }
        Node::ReduceMean(_) => {
            crate::ops::reduction::reduce_mean(node, values, device)?;
        }
        Node::ReduceMax(_) => {
            crate::ops::reduction::reduce_max(node, values, device)?;
        }
        Node::ReduceMin(_) => {
            crate::ops::reduction::reduce_min(node, values, device)?;
        }
        Node::ReduceProd(_) => {
            crate::ops::reduction::reduce_prod(node, values, device)?;
        }
        Node::ArgMax(_) => {
            crate::ops::reduction::argmax(node, values, device)?;
        }
        Node::ArgMin(_) => {
            crate::ops::reduction::argmin(node, values, device)?;
        }
        // Comparison operators
        Node::Equal(_) => {
            crate::ops::comparison::equal(node, values, device)?;
        }
        Node::Greater(_) => {
            crate::ops::comparison::greater(node, values, device)?;
        }
        Node::Less(_) => {
            crate::ops::comparison::less(node, values, device)?;
        }
        Node::GreaterOrEqual(_) => {
            crate::ops::comparison::greater_or_equal(node, values, device)?;
        }
        Node::LessOrEqual(_) => {
            crate::ops::comparison::less_or_equal(node, values, device)?;
        }
        Node::IsInf(_) => {
            crate::ops::comparison::is_inf(node, values, device)?;
        }
        Node::IsNaN(_) => {
            crate::ops::comparison::is_nan(node, values, device)?;
        }
        Node::And(_) => {
            crate::ops::comparison::and(node, values, device)?;
        }
        // Binary/variadic arithmetic operators
        Node::Pow(_) => {
            crate::ops::arithmetic::pow(node, values, device)?;
        }
        Node::Max(_) => {
            crate::ops::arithmetic::max_elementwise(node, values, device)?;
        }
        Node::Min(_) => {
            crate::ops::arithmetic::min_elementwise(node, values, device)?;
        }
        Node::Mod(_) => {
            crate::ops::arithmetic::modulo(node, values, device)?;
        }
        Node::Sum(_) => {
            crate::ops::arithmetic::sum_variadic(node, values, device)?;
        }
        Node::Mean(_) => {
            crate::ops::arithmetic::mean_variadic(node, values, device)?;
        }

        // Advanced data operations
        Node::Gather(_) => {
            crate::ops::advanced::gather(node, values, device)?;
        }
        Node::GatherElements(_) => {
            crate::ops::advanced::gather_elements(node, values, device)?;
        }
        Node::Where(_) => {
            crate::ops::advanced::where_op(node, values, device)?;
        }
        Node::TopK(_) => {
            crate::ops::advanced::topk(node, values, device)?;
        }
        Node::CumSum(_) => {
            crate::ops::advanced::cumsum(node, values, device)?;
        }

        // Additional shape manipulation operations
        Node::Split(_) => {
            crate::ops::shape::split(node, values, device)?;
        }
        Node::Slice(_) => {
            crate::ops::shape::slice(node, values, device)?;
        }
        Node::Expand(_) => {
            crate::ops::shape::expand(node, values, device)?;
        }
        Node::Tile(_) => {
            crate::ops::shape::tile(node, values, device)?;
        }
        Node::Pad(_) => {
            crate::ops::shape::pad(node, values, device)?;
        }

        Node::Cast(_) => {
            crate::ops::activation::cast(node, values, device)?;
        }
        Node::ConstantOfShape(_) => {
            crate::ops::activation::constant_of_shape(node, values, device)?;
        }

        // Spatial transform operations
        Node::Resize(_) => {
            crate::ops::shape::resize(node, values, device)?;
        }
        Node::DepthToSpace(_) => {
            crate::ops::shape::depth_to_space(node, values, device)?;
        }
        Node::SpaceToDepth(_) => {
            crate::ops::shape::space_to_depth(node, values, device)?;
        }

        // Advanced data operations - additional
        Node::OneHot(_) => {
            crate::ops::advanced::one_hot(node, values, device)?;
        }
        Node::NonZero(_) => {
            crate::ops::advanced::nonzero(node, values, device)?;
        }

        // Sequence operations
        Node::Range(_) => {
            crate::ops::sequence::range(node, values, device)?;
        }

        // ScatterND
        Node::ScatterND(_) => {
            crate::ops::advanced::scatter_nd(node, values, device)?;
        }

        // RNN operations
        Node::Lstm(_) => {
            crate::ops::rnn::lstm(node, values, device)?;
        }

        // Audio operations
        Node::Stft(_) => {
            crate::ops::audio::stft(node, values, device)?;
        }
        _ => anyhow::bail!("Unsupported operator: {} | {:?}", node.name(), node),
    }

    Ok(())
}
