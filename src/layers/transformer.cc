#include "ctranslate2/layers/transformer.h"

#include <cmath>
#include <cstring>

#include "ctranslate2/ops/dequantize.h"
#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/matmul.h"
#include "ctranslate2/ops/slide.h"
#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/ops/tanh.h"
#include "ctranslate2/ops/topk.h"

namespace ctranslate2 {
  namespace layers {

    FeedForwardNetwork::FeedForwardNetwork(const models::Model& model,
                                           const std::string& scope,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
      : _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _pre_norm(pre_norm)
      , _activation_type(activation_type)
      , _ff1(model, scope + "/linear_0", &_activation_type)
      , _ff1_noact(build_optional_layer<Dense>(model, scope + "/linear_0_noact"))
      , _ff2(model, scope + "/linear_1", nullptr, true)
      , _tensor_parallel(model.tensor_parallel()) {
    }

    void FeedForwardNetwork::operator()(const StorageView& input, StorageView& output) const {
      const StorageView* x = &input;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(input, output);
        x = &output;
      }

      const Device device = input.device();
      const DataType dtype = input.dtype();

      StorageView inner(dtype, device);
      _ff1(*x, inner);
      if (_ff1_noact) {
        StorageView linear(dtype, device);
        (*_ff1_noact)(*x, linear);
        ops::Mul()(linear, inner, inner);
      }

      _ff2(inner, output, _layer_norm ? &input : nullptr);

      if (_tensor_parallel) {
        Shape shape = output.shape();
        StorageView tmp(std::move(shape), output.dtype(), output.device());
        ops::ReduceAll red_op(ops::ReduceAll::RED_OP::SUM);
        red_op(output, tmp);
        output = std::move(tmp);
      }

      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type,
                                                     const bool use_flash_attention)
      : _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_feedforward_layer_norm"))
      , _ff(model, scope + "/ffn", pre_norm, activation_type) {
    }


    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView* lengths,
                                             StorageView& output,
                                             const Padder* padder,
                                             StorageView* position_bias) const {
      PROFILE("TransformerEncoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      // Check if using pre_post_layer_norm pattern (T5Gemma style)
      const bool pre_post_layer_norm = _input_layer_norm && _post_attention_layer_norm
                                        && _pre_feedforward_layer_norm && _post_feedforward_layer_norm;

      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);

        (*_input_layer_norm)(input, hidden);

        if (_self_attention)
          (*_self_attention)(hidden,
                          hidden,
                          lengths,
                          context,
                          nullptr,
                          nullptr,
                          nullptr,
                          padder,
                          padder,
                          true,
                          position_bias);

        // post_self_attn_layernorm
        (*_post_attention_layer_norm)(context, output);

        // residual + hidden_states
        ops::Add()(input, output, output);

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        // mlp
        _ff(hidden, output);

        // post_feedforward_layernorm
        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);

        // residual + hidden_states
        ops::Add()(context, output, output);
        return;
      }

      // Original path for standard pre-norm/post-norm architectures
      StorageView context(dtype, device);
      if (_self_attention)
        (*_self_attention)(input,
                        input,
                        lengths,
                        context,
                        nullptr,
                        nullptr,
                        nullptr,
                        padder,
                        padder,
                        true,
                        position_bias);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type,
                                                     const bool use_flash_attention,
                                                     Alibi* alibi)
      : _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true,
                        alibi)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true,
                        alibi)))
      , _shared_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/shared_layer_norm"))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_feedforward_layer_norm"))
      , _encoder_attention(build_optional_layer<MultiHeadAttention>(model,
                                                                    scope + "/attention",
                                                                    num_heads,
                                                                    /*self_attention=*/false,
                                                                    pre_norm,
                                                                    /*is_decoder=*/true))
      , _ff(model, scope + "/ffn", pre_norm, activation_type)
      , _external_pre_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_pre_encoder_attention_layer_norm"))
      , _external_post_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_post_encoder_attention_layer_norm"))
      , _kv_shared_layer_index(model.get_attribute_with_default<int32_t>(
                                     scope + "/kv_shared_layer_index", -1))
      , _layer_scalar(model.get_attribute_with_default<float>(scope + "/layer_scalar", 1.0f))
      , _per_layer_input_gate(build_optional_layer<Dense>(model, scope + "/per_layer_input_gate"))
      , _per_layer_projection(build_optional_layer<Dense>(model, scope + "/per_layer_projection"))
      , _post_per_layer_input_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_per_layer_input_norm"))
      , _router_norm(build_optional_layer<LayerNorm>(model, scope + "/router_norm"))
      , _router_scale_prescaled([&]() -> std::unique_ptr<const StorageView> {
          const auto* v = model.get_variable_if_exists(scope + "/router_scale");
          if (!v) return nullptr;
          const DataType dtype = get_default_float_type(model.effective_compute_type());
          return std::make_unique<StorageView>(v->to(dtype));
        }())
      , _router_proj([&]() -> std::unique_ptr<const Dense> {
          if (!model.get_variable_if_exists(scope + "/router_proj/weight")) return nullptr;
          return std::make_unique<Dense>(model, scope + "/router_proj");
        }())
      , _router_per_expert_scale([&]() -> std::unique_ptr<const StorageView> {
          const auto* v = model.get_variable_if_exists(scope + "/router_per_expert_scale");
          if (!v) return nullptr;
          const DataType dtype = get_default_float_type(model.effective_compute_type());
          return std::make_unique<StorageView>(v->to(dtype));
        }())
      , _experts_gate_up_proj([&]() -> std::unique_ptr<const StorageView> {
          const auto* v = model.get_variable_if_exists(scope + "/experts_gate_up_proj");
          if (!v) return nullptr;
          const DataType dtype = get_default_float_type(model.effective_compute_type());
          return std::make_unique<StorageView>(v->to(dtype));
        }())
      , _experts_down_proj([&]() -> std::unique_ptr<const StorageView> {
          const auto* v = model.get_variable_if_exists(scope + "/experts_down_proj");
          if (!v) return nullptr;
          const DataType dtype = get_default_float_type(model.effective_compute_type());
          return std::make_unique<StorageView>(v->to(dtype));
        }())
      , _moe_top_k(model.get_attribute_with_default<int32_t>(scope + "/moe_top_k", 0))
      , _post_feedforward_layernorm_1(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_feedforward_layernorm_1"))
      , _pre_feedforward_layernorm_2(build_optional_layer<LayerNorm>(
                                     model, scope + "/pre_feedforward_layernorm_2"))
      , _post_feedforward_layernorm_2(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_feedforward_layernorm_2"))
      {
    }

    void TransformerDecoderLayer::operator()(const StorageView& input,
                                             const StorageView* input_length,
                                             const StorageView* memory,
                                             const StorageView* memory_lengths,
                                             StorageView* cached_self_attn_keys,
                                             StorageView* cached_self_attn_values,
                                             StorageView* cached_attn_keys,
                                             StorageView* cached_attn_values,
                                             StorageView& output,
                                             StorageView* attention,
                                             const Padder* input_padder,
                                             const Padder* memory_padder,
                                             bool return_normalized_attention,
                                             StorageView* position_bias,
                                             dim_t offset,
                                             const StorageView* per_layer_input,
                                             bool kv_read_only) const {
      PROFILE("TransformerDecoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      const bool pre_post_layer_norm = _post_feedforward_layer_norm && _pre_feedforward_layer_norm;
      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);
        (*_input_layer_norm)(input, hidden);

        if (_self_attention)
          (*_self_attention)(hidden,
                             hidden,
                             input_length,
                             context,
                             cached_self_attn_keys,
                             cached_self_attn_values,
                             nullptr,
                             input_padder,
                             input_padder,
                             true,
                             position_bias,
                             offset,
                             kv_read_only);

        (*_post_attention_layer_norm)(context, output);
        ops::Add()(output, input, output);

        if (_encoder_attention) {
            StorageView cross_attn_in = output;  // save for residual

            StorageView query_normalized(dtype, device);
            if (_external_pre_encoder_attention_layer_norm) {
                (*_external_pre_encoder_attention_layer_norm)(output, query_normalized);
            }
            else {
                query_normalized.shallow_copy(output);
            }

            (*_encoder_attention)(query_normalized,
                                  *memory,
                                  memory_lengths,
                                  context,
                                  cached_attn_keys,
                                  cached_attn_values,
                                  attention,
                                  input_padder,
                                  memory_padder,
                                  return_normalized_attention);

            if (_external_post_encoder_attention_layer_norm) {
                (*_external_post_encoder_attention_layer_norm)(context, context);
            }
            ops::Add()(context, cross_attn_in, output);
        }

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        _ff(hidden, output);

        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);
        ops::Add()(output, context, output);

        // Gemma 4: Dense FFN path done. Apply PLE block if present.
        if (per_layer_input && _per_layer_input_gate && _per_layer_projection
            && _post_per_layer_input_norm) {
          // gate(output) → GELUTanh → * per_layer_input → projection → post_norm → + output
          StorageView gate_out(dtype, device);
          (*_per_layer_input_gate)(output, gate_out);
          { const ops::GELU gelu_tanh(ops::GELU::Approximation::Tanh); gelu_tanh(gate_out, gate_out); }
          // Broadcast per_layer_input [B, T, D] to match gate_out [B, T, D]
          ops::Mul()(gate_out, *per_layer_input, gate_out);
          StorageView proj_out(dtype, device);
          (*_per_layer_projection)(gate_out, proj_out);
          StorageView normed(dtype, device);
          (*_post_per_layer_input_norm)(proj_out, normed);
          ops::Add()(output, normed, output);
        }

        // Gemma 4 MoE: parallel sparse expert path (placeholder — full impl in gemma4.cc)
        // This path is not reached for E2B (dense-only); 26B-A4B MoE support is deferred.
        (void)_router_norm; (void)_router_proj; (void)_experts_gate_up_proj;
        (void)_experts_down_proj; (void)_router_scale_prescaled;
        (void)_router_per_expert_scale; (void)_pre_feedforward_layernorm_2;
        (void)_post_feedforward_layernorm_2;

        // Gemma 4: multiply full hidden state by layer_scalar
        // Note: scalar StorageView must be on CPU; primitives<CUDA>::mul(scalar, ...) reads the
        // value on the host via b.data<T>()[0], so a CUDA scalar would segfault.
        if (_layer_scalar != 1.0f)
          ops::Mul()(output, StorageView(_layer_scalar), output);

        return;
      }

      const bool use_parallel_residual = _shared_layer_norm || _input_layer_norm;

      if (use_parallel_residual) {
        // The parallel residual implementation assumes there is no cross attention.
        StorageView hidden(dtype, device);

        if (_shared_layer_norm)
          (*_shared_layer_norm)(input, hidden);
        else
          (*_input_layer_norm)(input, hidden);

        StorageView attn(dtype, device);
        if (_self_attention)
          (*_self_attention)(hidden,
                        hidden,
                        input_length,
                        attn,
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        nullptr,
                        input_padder,
                        input_padder,
                        true,
                        position_bias,
                        offset);

        if (_post_attention_layer_norm)
          (*_post_attention_layer_norm)(input, hidden);

        _ff(hidden, output);

        ops::Add()(output, input, output);
        ops::Add()(output, attn, output);

        return;
      }
      if (_self_attention)
        (*_self_attention)(input,
                      input,
                      input_length,
                      output,
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      nullptr,
                      input_padder,
                      input_padder,
                      true,
                      position_bias,
                      offset);

      StorageView context(dtype, device);
      if (_encoder_attention) {
        (*_encoder_attention)(output,
                              *memory,
                              memory_lengths,
                              context,
                              cached_attn_keys,
                              cached_attn_values,
                              attention,
                              input_padder,
                              memory_padder,
                              return_normalized_attention);
      }
      else {
        context = std::move(output);
      }

      _ff(context, output);
    }


    static std::unique_ptr<PositionEncoder>
    build_position_encoder(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      if (model.get_variable_if_exists(scope + "/encodings"))
        return std::make_unique<PositionEmbedding>(model, scope);
      else
        return std::make_unique<SinusoidalPositionEncoder>(embeddings.output_size(),
                                                           embeddings.output_type(),
                                                           model.device());
    }

    static std::unique_ptr<const StorageView>
    build_embeddings_scale(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      const auto* scale = model.get_variable_if_exists(scope + "/scale_embeddings");

      // Backward compatibility with older models.
      if (!scale)
        scale = model.get_variable_if_exists(scope + "/embeddings/multiply_by_sqrt_depth");

      StorageView value;

      // The attribute can either be a boolean flag or the actual scale value.
      if (!scale || (scale->dtype() == DataType::INT8 && scale->as_scalar<int8_t>()))
        value = StorageView(std::sqrt(static_cast<float>(embeddings.output_size())));
      else if (scale->dtype() != DataType::INT8 && scale->as_scalar<float>() != 1.f)
        value = *scale;
      else
        return nullptr;

      return std::make_unique<StorageView>(value.to(embeddings.output_type()));
    }


    TransformerEncoder::TransformerEncoder(const models::Model& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings",
                    model.get_enum_value<EmbeddingsMerge>(scope + "/embeddings_merge"))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _use_flash_attention(model.use_flash_attention())
      , _layers(build_layers_list<const TransformerEncoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation")))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _tensor_parallel(model.tensor_parallel())
    {
    }

    void TransformerEncoder::operator()(const std::vector<StorageView>& ids,
                                        const StorageView* lengths,
                                        StorageView& output) {
      PROFILE("TransformerEncoder");
      StorageView input(output.dtype(), output.device());
      _embeddings(ids, input);
      if (_embeddings_scale)
        ops::Mul()(input, *_embeddings_scale, input);
      if (_position_encoder)
        (*_position_encoder)(input);
      if (_layernorm_embedding)
        (*_layernorm_embedding)(input, input);

      const dim_t max_time = input.dim(1);

      // Remove padding to reduce the amount of computation.
      std::unique_ptr<Padder> padder;
      std::unique_ptr<StorageView> lengths_mask;

      if (lengths) {
        if (Padder::allow_padding_removal(output.device(), _compute_type)) {
          padder = std::make_unique<Padder>(*lengths, max_time);
          padder->remove_padding(input);
        }

        int num_heads = _num_heads;
        if (_tensor_parallel) {
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
        }
        lengths_mask = std::make_unique<StorageView>(
          layers::MultiHeadAttention::prepare_length_mask(*lengths, num_heads, max_time));
      }

      StorageView position_bias(output.dtype(), output.device());

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths_mask.get(), output, padder.get(), &position_bias);
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
      if (_output_norm)
        (*_output_norm)(output, output);
      if (padder)
        padder->add_padding(output);
    }


    static std::unique_ptr<Alibi> make_alibi(const models::Model& model, const std::string& scope) {
      const bool use_alibi = model.get_flag_with_default(scope + "/alibi", false);
      if (!use_alibi)
        return nullptr;

      const bool use_positive_positions = model.get_flag_with_default(
        scope + "/alibi_use_positive_positions", true);
      const bool scale_alibi = model.get_flag_with_default(
        scope + "/scale_alibi", false);

      return std::make_unique<Alibi>(use_positive_positions, scale_alibi);
    }

    TransformerDecoder::TransformerDecoder(const models::Model& model, const std::string& scope)
      : Decoder(model.device())
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _start_from_zero_embedding(model.get_flag_with_default(scope + "/start_from_zero_embedding",
                                                               false))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _project_in(build_optional_layer<Dense>(model, scope + "/project_in"))
      , _project_out(build_optional_layer<Dense>(model, scope + "/project_out"))
      , _alibi(make_alibi(model, scope))
      , _use_flash_attention(model.use_flash_attention())
      , _layers(build_layers_list<const TransformerDecoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation"),
                  _use_flash_attention,
                  _alibi.get()))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _with_encoder_attention(_layers.front()->has_cross_attention())
      , _proj(model, scope + "/projection")
      , _sliding_window(model.get_attribute_with_default<int32_t>(scope + "/sliding_window", 0))
      , _tensor_parallel(model.tensor_parallel())
      , _final_logit_softcap(model.get_attribute_with_default<float>(scope + "/final_logit_softcap", 0.0f))
      , _per_layer_embeddings([&]() -> std::unique_ptr<const StorageView> {
          const auto* v = model.get_variable_if_exists(scope + "/per_layer_token_embedding");
          if (!v) return nullptr;
          // Keep as int8 on CPU for efficient lookup
          return std::make_unique<StorageView>(v->to(Device::CPU));
        }())
      , _per_layer_embedding_scales([&]() -> std::unique_ptr<const StorageView> {
          const auto* v = model.get_variable_if_exists(scope + "/per_layer_token_scale");
          if (!v) return nullptr;
          return std::make_unique<StorageView>(v->to(DataType::FLOAT32).to(Device::CPU));
        }())
      , _per_layer_model_proj(build_optional_layer<Dense>(model, scope + "/per_layer_model_projection"))
      , _per_layer_proj_norm(build_optional_layer<LayerNorm>(model, scope + "/per_layer_projection_norm")) {

      dim_t alignment_layer = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_layer", -1));
      dim_t alignment_heads = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_heads", 1));

      if (alignment_layer < 0)
        alignment_layer = _layers.size() + alignment_layer;
      if (alignment_heads == 0)
        alignment_heads = _num_heads;

      set_alignment_heads(alignment_layer, alignment_heads);

      const auto* outputs_scale = model.get_variable_if_exists(scope + "/scale_outputs");
      if (outputs_scale) {
        const DataType dtype = get_default_float_type(_compute_type);
        _outputs_scale = std::make_unique<StorageView>(outputs_scale->to(dtype));
      }
    }

    DecoderState TransformerDecoder::initial_state(bool iterative_decoding) const {
      DecoderState state;

      if (iterative_decoding) {
        const size_t state_size = _layers.size() * (_with_encoder_attention ? 4 : 2);
        state.reserve(state_size);

        const DataType dtype = output_type();

        for (size_t i = 0; i < _layers.size(); ++i) {
          const std::string i_str = std::to_string(i);
          state.emplace("self_keys_" + i_str, StorageView(dtype, _device));
          state.emplace("self_values_" + i_str, StorageView(dtype, _device));
          if (_with_encoder_attention) {
            state.emplace("memory_keys_" + i_str, StorageView(dtype, _device));
            state.emplace("memory_values_" + i_str, StorageView(dtype, _device));
          }
        }
      }

      return state;
    }

    bool TransformerDecoder::replicate_state(const std::string& name) const {
      // No need to replicate projected memory keys and values as they are the same for each beam.
      return !_with_encoder_attention || !starts_with(name, "memory");
    }

    void TransformerDecoder::set_alignment_heads(const dim_t layer,
                                                 const dim_t num_heads_to_average) {
      std::vector<dim_t> range(num_heads_to_average);
      std::iota(range.begin(), range.end(), dim_t(0));

      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      _alignment_heads[layer] = std::move(range);

      _average_alignment_heads = true;
    }

    void TransformerDecoder::set_alignment_heads(const std::vector<std::pair<dim_t, dim_t>>& alignment_heads) {
      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      for (const auto& [layer, head] : alignment_heads)
        _alignment_heads[layer].push_back(head);

      _average_alignment_heads = false;
    }

    std::unique_ptr<StorageView>
    TransformerDecoder::get_layer_alignment_heads(const dim_t layer, const dim_t batch_size) const {
      if (_alignment_heads.empty())
        return nullptr;

      const auto& heads = _alignment_heads[layer];
      const dim_t num_heads = heads.size();

      if (heads.empty())
        return nullptr;

      std::vector<int32_t> indices;
      indices.reserve(batch_size * num_heads);
      for (dim_t i = 0; i < batch_size; ++i)
        indices.insert(indices.end(), heads.begin(), heads.end());

      return std::make_unique<StorageView>(Shape{batch_size, num_heads}, indices, _device);
    }

    void TransformerDecoder::operator()(dim_t step,
                                        const StorageView& ids,
                                        DecoderState& state,
                                        StorageView* logits,
                                        StorageView* attention) {
      return decode(ids, nullptr, step, state, logits, attention);
    }

    void TransformerDecoder::operator()(const StorageView& ids,
                                        const StorageView& lengths,
                                        DecoderState& state,
                                        StorageView& logits,
                                        StorageView* attention) {
      return decode(ids, &lengths, -1, state, &logits, attention);
    }

    // -----------------------------------------------------------------------
    // _apply_output_projection: shared output-projection block.
    //
    // Applies, in order: output_norm (if present), project_out (if present),
    // outputs_scale (if present), and the final projection (or hidden-state
    // passthrough when return_logits == false).
    //
    // Both decode() and decode_from_embeds() end with identical output logic,
    // so they delegate here to avoid maintaining the same code twice.
    // -----------------------------------------------------------------------
    void TransformerDecoder::_apply_output_projection(StorageView& layer_in,
                                                       StorageView& layer_out,
                                                       StorageView* outputs,
                                                       bool return_logits,
                                                       bool is_sequence,
                                                       const Padder* input_padder) {
      if (!outputs)
        return;

      if (_output_norm)
        (*_output_norm)(layer_in, layer_in);
      if (_project_out) {
        (*_project_out)(layer_in, layer_out);
        layer_in = std::move(layer_out);
      }
      if (_outputs_scale)
        ops::Mul()(layer_in, *_outputs_scale, layer_in);

      if (return_logits) {
        _proj(layer_in, *outputs);
        // Gemma 4: final logit soft-capping: tanh(x/cap)*cap
        // Scalars must be CPU StorageViews (see CUDA scalar deref note in layer_scalar fix).
        if (_final_logit_softcap > 0.f) {
          ops::Mul()(*outputs, StorageView(1.f / _final_logit_softcap), *outputs);
          ops::Tanh()(*outputs, *outputs);
          ops::Mul()(*outputs, StorageView(_final_logit_softcap), *outputs);
        }
      } else {
        *outputs = std::move(layer_in);
      }

      if (!is_sequence)
        outputs->squeeze(1);
      else if (input_padder)
        input_padder->add_padding(*outputs);
    }

    void TransformerDecoder::decode(const StorageView& ids,
                                    const StorageView* lengths,
                                    dim_t step,
                                    DecoderState& state,
                                    StorageView* outputs,
                                    StorageView* attention,
                                    bool return_logits) {
      PROFILE("TransformerDecoder");
      const DataType dtype = output_type();
      const Device device = ids.device();
      const bool is_sequence = ids.rank() > 1;

      StorageView layer_in(dtype, device);
      StorageView layer_out(dtype, device);

      _embeddings(ids, layer_in);
      if (_start_from_zero_embedding)
        zero_first_timestep(layer_in, step);
      if (_embeddings_scale && (!_start_from_zero_embedding || step != 0))
        ops::Mul()(layer_in, *_embeddings_scale, layer_in);
      if (_project_in) {
        (*_project_in)(layer_in, layer_out);
        layer_in = std::move(layer_out);
      }
      if (layer_in.rank() == 2)
        layer_in.expand_dims(1);
      if (_position_encoder)
        (*_position_encoder)(layer_in, std::max(step, dim_t(0)));
      if (_layernorm_embedding)
        (*_layernorm_embedding)(layer_in, layer_in);

      const dim_t batch_size = layer_in.dim(0);
      dim_t max_time;

      if (_sliding_window > 0 && layer_in.dim(1) > _sliding_window) {
        max_time = _sliding_window;
      } else
        max_time = layer_in.dim(1);

      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths;
      std::unique_ptr<const StorageView> input_lengths_mask;

      if (is_sequence && !lengths) {
        input_lengths = std::make_unique<StorageView>(Shape{ids.dim(0)}, int32_t(max_time), device);
        lengths = input_lengths.get();
      }

      bool multi_query = _layers.front()->get_self_attention().multi_query();

      if (lengths) {
        if (allow_padding_removal) {
          input_padder = std::make_unique<Padder>(*lengths, max_time);
          input_padder->remove_padding(layer_in);
        }

        dim_t num_heads = _num_heads;
        if (_tensor_parallel) {
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
        }

        StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
          *lengths,
          num_heads,
          max_time,
          /*mask_future=*/true,
          multi_query);


        if (step > 0)
          ops::Add()(lengths_mask, StorageView(int32_t(step)), lengths_mask);

        input_lengths_mask = std::make_unique<StorageView>(std::move(lengths_mask));
      }

      StorageView* memory = nullptr;
      std::unique_ptr<const StorageView> memory_lengths_mask;
      std::unique_ptr<const Padder> memory_padder;
      if (_with_encoder_attention) {
        const auto it = state.find("memory_lengths");
        const StorageView* memory_lengths = it != state.end() ? &it->second : nullptr;

        if (step <= 0) {
          memory = &state.at("memory");

          if (memory_lengths && allow_padding_removal) {
            memory_padder = std::make_unique<Padder>(*memory_lengths, memory->dim(1));
            memory_padder->remove_padding(*memory);
          }
        }

        if (memory_lengths) {
          dim_t num_heads = _num_heads;
          if (_tensor_parallel) {
            num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
          }
          const dim_t beam_size = batch_size / memory_lengths->dim(0);
          memory_lengths_mask = std::make_unique<StorageView>(
            layers::MultiHeadAttention::prepare_length_mask(*memory_lengths,
                                                            num_heads,
                                                            beam_size > 1 ? beam_size : max_time));
        }
      }

      std::vector<StorageView> alignment_heads;
      if (attention)
        alignment_heads.reserve(_layers.size());

      StorageView position_bias(dtype, device);

      // Gemma 4 E-series: compute combined per-layer inputs from ids + layer_in.
      const std::vector<StorageView> per_layer_embs =
          _compute_per_layer_inputs(&ids, layer_in);

      std::vector<StorageView> layer_ins;

      while (true) {
        dim_t prompt_size = layer_in.dim(1);
        if (_sliding_window == 0 || prompt_size <= _sliding_window || _use_flash_attention) {
          layer_ins.push_back(std::move(layer_in));
          break;
        }
        if (layer_in.dim(1) > _sliding_window) {
          StorageView tmp(dtype, device);
          const ops::Split split_op(1, {_sliding_window, prompt_size - _sliding_window});
          split_op(layer_in, tmp, layer_in);
          layer_ins.push_back(std::move(tmp));
        }
      }

      for (size_t i = 0; i < layer_ins.size(); ++i) {
        StorageView* layer_in_chunk = &layer_ins[i];
        for (size_t l = 0; l < _layers.size(); ++l) {
          StorageView* cached_self_attn_keys = nullptr;
          StorageView* cached_self_attn_values = nullptr;
          StorageView* cached_attn_keys = nullptr;
          StorageView* cached_attn_values = nullptr;

          bool kv_read_only = false;
          if (step >= 0) {
            const dim_t src_idx = _layers[l]->kv_shared_layer_index();
            if (src_idx >= 0) {
              const std::string src_str = std::to_string(src_idx);
              cached_self_attn_keys   = &state.at("self_keys_"   + src_str);
              cached_self_attn_values = &state.at("self_values_" + src_str);
              kv_read_only = true;
            } else {
              const std::string l_str = std::to_string(l);
              cached_self_attn_keys   = &state.at("self_keys_"   + l_str);
              cached_self_attn_values = &state.at("self_values_" + l_str);
            }
            if (_with_encoder_attention) {
              const std::string l_str = std::to_string(l);
              cached_attn_keys   = &state.at("memory_keys_"   + l_str);
              cached_attn_values = &state.at("memory_values_" + l_str);
            }
          }

          std::unique_ptr<StorageView> heads_to_select = get_layer_alignment_heads(l, batch_size);
          std::unique_ptr<StorageView> layer_attention;
          if (attention && heads_to_select)
            layer_attention = std::make_unique<StorageView>(dtype, device);

          dim_t offset = _sliding_window * i + step;
          offset = offset < 0 ? 0 : offset;
          if (i > 0) {
            auto max_tokens = _sliding_window + layer_in_chunk->dim(1);
            StorageView tmp_lengths = StorageView(Shape{layer_in_chunk->dim(0)}, int32_t(max_tokens), device);
            int num_heads = _num_heads;
            if (_tensor_parallel) {
              num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
            }
            StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
              tmp_lengths,
              num_heads,
              max_tokens,
              /*mask_future=*/true,
              multi_query);

            const ops::Slide slide_lengths_op(2, _sliding_window, layer_in_chunk->dim(1));
            // reuse tmp_lengths
            slide_lengths_op(lengths_mask, tmp_lengths);
            input_lengths_mask = std::make_unique<StorageView>(std::move(tmp_lengths));
          }

          const StorageView* ple = !per_layer_embs.empty()
                                   ? &per_layer_embs[l] : nullptr;
          (*_layers[l])(*layer_in_chunk,
                        input_lengths_mask.get(),
                        memory,
                        memory_lengths_mask.get(),
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        cached_attn_keys,
                        cached_attn_values,
                        layer_out,
                        layer_attention.get(),
                        input_padder.get(),
                        memory_padder.get(),
                        return_normalized_attention(),
                        &position_bias,
                        offset,
                        ple,
                        kv_read_only);
          *layer_in_chunk = std::move(layer_out);

          if (layer_attention) {
            alignment_heads.emplace_back(dtype, device);
            ops::Gather(1, 1)(*layer_attention, *heads_to_select, alignment_heads.back());
          }
        }
        layer_in = std::move(*layer_in_chunk);
      }

      if (step == 0) {
        // The memory is no longer needed as its projections were cached in the first step.
        state.erase("memory");
      }

      if (attention && !alignment_heads.empty()) {
        if (_average_alignment_heads) {
          ops::Mean(1)(alignment_heads[0], *attention);
          if (!is_sequence)
            attention->squeeze(1);

        } else {
          std::vector<const StorageView*> alignment_heads_ptr;
          alignment_heads_ptr.reserve(alignment_heads.size());
          for (const auto& heads : alignment_heads)
            alignment_heads_ptr.emplace_back(&heads);

          ops::Concat(1)(alignment_heads_ptr, *attention);
          if (!is_sequence)
            attention->squeeze(2);
        }
      }

      _apply_output_projection(layer_in, layer_out, outputs, return_logits,
                               is_sequence, input_padder.get());
    }

    // -----------------------------------------------------------------------
    // decode_from_embeds: like decode() but accepts pre-computed embeddings.
    // input_ids (optional): used for Gemma 4 E-series token-side PLE lookup.
    // -----------------------------------------------------------------------
    void TransformerDecoder::decode_from_embeds(const StorageView& inputs_embeds,
                                                const StorageView* input_ids,
                                                const StorageView* lengths,
                                                dim_t step,
                                                DecoderState& state,
                                                StorageView* outputs,
                                                StorageView* attention,
                                                bool return_logits) {
      PROFILE("TransformerDecoder::decode_from_embeds");
      const DataType dtype = output_type();
      const Device device = inputs_embeds.device();
      constexpr bool is_sequence = true;  // inputs_embeds is always (batch, seq, hidden)

      StorageView layer_in(dtype, device);
      StorageView layer_out(dtype, device);

      // Use pre-computed embeddings directly, cast to model compute dtype.
      layer_in = inputs_embeds.to(dtype);

      // Position encoding: for Qwen2 RoPE is inside attention layers, so typically nullptr.
      // Apply if present (e.g. sinusoidal PE models).
      if (_position_encoder)
        (*_position_encoder)(layer_in, std::max(step, dim_t(0)));
      if (_layernorm_embedding)
        (*_layernorm_embedding)(layer_in, layer_in);

      // inputs_embeds is already rank 3 — no need for expand_dims.

      const dim_t batch_size = layer_in.dim(0);
      const dim_t max_time = layer_in.dim(1);

      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths;
      std::unique_ptr<const StorageView> input_lengths_mask;

      if (!lengths) {
        input_lengths = std::make_unique<StorageView>(
          Shape{layer_in.dim(0)}, int32_t(max_time), device);
        lengths = input_lengths.get();
      }

      bool multi_query = _layers.front()->get_self_attention().multi_query();

      if (lengths) {
        if (allow_padding_removal) {
          input_padder = std::make_unique<Padder>(*lengths, max_time);
          input_padder->remove_padding(layer_in);
        }

        dim_t num_heads = _num_heads;
        if (_tensor_parallel)
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());

        StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
          *lengths, num_heads, max_time, /*mask_future=*/true, multi_query);

        if (step > 0)
          ops::Add()(lengths_mask, StorageView(int32_t(step)), lengths_mask);

        input_lengths_mask = std::make_unique<StorageView>(std::move(lengths_mask));
      }

      std::vector<StorageView> alignment_heads;
      if (attention)
        alignment_heads.reserve(_layers.size());

      StorageView position_bias(dtype, device);
      std::vector<StorageView> layer_ins;
      layer_ins.push_back(std::move(layer_in));

      // Gemma 4 E-series: compute combined per-layer inputs.
      const std::vector<StorageView> per_layer_embs =
          _compute_per_layer_inputs(input_ids, inputs_embeds.to(dtype));

      for (size_t i = 0; i < layer_ins.size(); ++i) {
        StorageView* layer_in_chunk = &layer_ins[i];
        for (size_t l = 0; l < _layers.size(); ++l) {
          StorageView* cached_self_attn_keys = nullptr;
          StorageView* cached_self_attn_values = nullptr;

          bool kv_read_only = false;
          if (step >= 0) {
            const dim_t src_idx = _layers[l]->kv_shared_layer_index();
            if (src_idx >= 0) {
              const std::string src_str = std::to_string(src_idx);
              cached_self_attn_keys   = &state.at("self_keys_"   + src_str);
              cached_self_attn_values = &state.at("self_values_" + src_str);
              kv_read_only = true;
            } else {
              const std::string l_str = std::to_string(l);
              cached_self_attn_keys   = &state.at("self_keys_"   + l_str);
              cached_self_attn_values = &state.at("self_values_" + l_str);
            }
          }

          std::unique_ptr<StorageView> heads_to_select = get_layer_alignment_heads(l, batch_size);
          std::unique_ptr<StorageView> layer_attention;
          if (attention && heads_to_select)
            layer_attention = std::make_unique<StorageView>(dtype, device);

          const dim_t offset = std::max(dim_t(0), dim_t(_sliding_window) * dim_t(i) + step);

          const StorageView* ple = !per_layer_embs.empty()
                                   ? &per_layer_embs[l] : nullptr;
          (*_layers[l])(*layer_in_chunk,
                        input_lengths_mask.get(),
                        /*memory=*/nullptr,
                        /*memory_lengths_mask=*/nullptr,
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        /*cached_attn_keys=*/nullptr,
                        /*cached_attn_values=*/nullptr,
                        layer_out,
                        layer_attention.get(),
                        input_padder.get(),
                        /*memory_padder=*/nullptr,
                        return_normalized_attention(),
                        &position_bias,
                        offset,
                        ple,
                        kv_read_only);
          *layer_in_chunk = std::move(layer_out);

          if (layer_attention) {
            alignment_heads.emplace_back(dtype, device);
            ops::Gather(1, 1)(*layer_attention, *heads_to_select, alignment_heads.back());
          }
        }
        layer_in = std::move(*layer_in_chunk);
      }

      // Always a sequence — is_sequence=true, input_padder may or may not be set.
      _apply_output_projection(layer_in, layer_out, outputs, return_logits,
                               /*is_sequence=*/true, input_padder.get());
    }

    // -----------------------------------------------------------------------
    // decode_from_embeds: backward-compatible overload (no input_ids).
    // -----------------------------------------------------------------------
    void TransformerDecoder::decode_from_embeds(const StorageView& inputs_embeds,
                                                const StorageView* lengths,
                                                dim_t step,
                                                DecoderState& state,
                                                StorageView* outputs,
                                                StorageView* attention,
                                                bool return_logits) {
      decode_from_embeds(inputs_embeds, /*input_ids=*/nullptr,
                         lengths, step, state, outputs, attention, return_logits);
    }

    // -----------------------------------------------------------------------
    // _compute_per_layer_inputs: Gemma 4 E-series PLE computation.
    //
    // Computes the combined per-layer input tensor for all layers:
    //   model_ple = per_layer_model_projection(inputs_embeds) * hidden_size^{-0.5}
    //   model_ple = per_layer_projection_norm(model_ple)  [B, T, num_L, D]
    //   token_ple = embed_tokens_per_layer[ids] * sqrt(D)  [B, T, num_L, D] (if ids)
    //   combined  = (model_ple + token_ple) * 2^{-0.5}     (or just model_ple)
    //
    // Returns a vector of num_layers StorageViews, each [batch, seq, per_layer_dim].
    // Returns empty vector if PLE weights are not loaded.
    // -----------------------------------------------------------------------
    std::vector<StorageView> TransformerDecoder::_compute_per_layer_inputs(
        const StorageView* ids,
        const StorageView& inputs_embeds) const {

      if (!_per_layer_model_proj || !_per_layer_proj_norm)
        return {};

      const DataType dtype = output_type();
      const Device device = inputs_embeds.device();
      const dim_t B = inputs_embeds.dim(0);
      const dim_t T = inputs_embeds.dim(1);
      const dim_t H = inputs_embeds.dim(2);
      const dim_t num_layers = static_cast<dim_t>(_layers.size());

      // Step 1: model projection: [B, T, H] → [B, T, num_layers * per_layer_dim]
      StorageView model_ple(dtype, device);
      {
        StorageView embeds_cast = inputs_embeds.to(dtype);
        (*_per_layer_model_proj)(embeds_cast, model_ple);
      }

      // Scale by hidden_size^{-0.5}
      {
        const float scale = 1.0f / std::sqrt(static_cast<float>(H));
        ops::Mul()(model_ple, StorageView(scale), model_ple);  // CPU scalar (CUDA safe)
      }

      const dim_t ple_total = model_ple.dim(2);
      const dim_t per_layer_dim = ple_total / num_layers;

      // Step 2: reshape to [B*T*num_layers, per_layer_dim] and apply norm.
      model_ple.reshape({B * T * num_layers, per_layer_dim});
      StorageView normed_ple(dtype, device);
      (*_per_layer_proj_norm)(model_ple, normed_ple);
      normed_ple.reshape({B, T, num_layers, per_layer_dim});

      // Step 3: token PLE lookup (if ids and embedding table are available).
      StorageView combined_ple(dtype, device);
      if (ids && _per_layer_embeddings) {
        const StorageView ids_cpu = ids->to(Device::CPU);
        const bool is_seq = (ids_cpu.rank() == 2);
        const dim_t n_ids = static_cast<dim_t>(ids_cpu.size());

        StorageView flat_ids(ids_cpu);
        flat_ids.reshape({n_ids});

        StorageView token_ple_dev(dtype, device);
        if (_per_layer_embeddings->dtype() == DataType::INT8) {
          // Legacy INT8 path: gather + manual dequantize with per-row scale.
          StorageView token_ple_int8(DataType::INT8, Device::CPU);
          ops::Gather()(*_per_layer_embeddings, flat_ids, token_ple_int8);

          StorageView scales_gathered(DataType::FLOAT32, Device::CPU);
          ops::Gather()(*_per_layer_embedding_scales, flat_ids, scales_gathered);

          const dim_t n_rows = static_cast<dim_t>(n_ids);
          const dim_t n_cols = static_cast<dim_t>(token_ple_int8.dim(1));
          StorageView token_ple_float(DataType::FLOAT32, Device::CPU);
          token_ple_float.resize({n_rows, n_cols});

          const int8_t* src_data   = token_ple_int8.data<int8_t>();
          const float*  scale_data = scales_gathered.data<float>();
          float*        dst_data   = token_ple_float.data<float>();
          for (dim_t i = 0; i < n_rows; ++i) {
            const float s = scale_data[i];
            for (dim_t j = 0; j < n_cols; ++j)
              dst_data[i * n_cols + j] = static_cast<float>(src_data[i * n_cols + j]) * s;
          }
          token_ple_dev = token_ple_float.to(device).to(dtype);
        } else {
          // Float16 path: gather directly then cast to compute dtype.
          // The embedding table is stored as float16, matching HF's bfloat16→float16
          // cast exactly — no quantization error.
          StorageView token_ple_cpu(_per_layer_embeddings->dtype(), Device::CPU);
          ops::Gather()(*_per_layer_embeddings, flat_ids, token_ple_cpu);
          token_ple_dev = token_ple_cpu.to(device).to(dtype);
        }

        // Scale by sqrt(per_layer_dim)
        {
          const float embed_scale = std::sqrt(static_cast<float>(per_layer_dim));
          ops::Mul()(token_ple_dev, StorageView(embed_scale), token_ple_dev);  // CPU scalar (CUDA safe)
        }

        // Reshape to [B, T, num_layers, per_layer_dim]
        if (is_seq) {
          token_ple_dev.reshape({B, T, num_layers, per_layer_dim});
        } else {
          token_ple_dev.reshape({B, 1, num_layers, per_layer_dim});
        }

        // Combine: (model_ple + token_ple) * 2^{-0.5}
        StorageView added(dtype, device);
        ops::Add()(normed_ple, token_ple_dev, added);
        ops::Mul()(added, StorageView(0.70710678f), combined_ple);  // CPU scalar (CUDA safe)
      } else {
        combined_ple = std::move(normed_ple);
      }

      // Step 4: slice into per-layer tensors [B, T, per_layer_dim].
      const dim_t T_actual = combined_ple.dim(1);
      std::vector<StorageView> result(static_cast<size_t>(num_layers),
                                      StorageView(dtype, device));
      for (dim_t l = 0; l < num_layers; ++l) {
        StorageView tmp(combined_ple);
        tmp.reshape({B * T_actual, num_layers, per_layer_dim});
        ops::Slide(1, l, 1)(tmp, result[static_cast<size_t>(l)]);
        result[static_cast<size_t>(l)].reshape({B, T_actual, per_layer_dim});
      }

      return result;
    }

  }
}
