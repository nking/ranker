#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

//Register the Op's Interface (Inputs and Outputs)
REGISTER_OP("BuildEnrichedGraph")
    .Input("user_ids: int32")
    .Input("timestamps: int64")
    .Input("candidate_ids: int32")
    .Input("candidate_embeddings: float32")
    .Input("user_embeddings: float32")

    .Attr("max_graphs: int >= 1")
    .Attr("max_nodes: int >= 1")
    .Attr("max_edges: int >= 1")

    .Output("n_node: int32")
    .Output("n_edge: int32")
    .Output("senders: int32")
    .Output("receivers: int32")
    .Output("edge_features: int32")
    .Output("node_ids: int32")
    .Output("node_labels: int32")
    .Output("node_types: int32")
    .Output("candidate_mask: bool")

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int max_graphs, max_nodes, max_edges;
      TF_RETURN_IF_ERROR(c->GetAttr("max_graphs", &max_graphs));
      TF_RETURN_IF_ERROR(c->GetAttr("max_nodes", &max_nodes));
      TF_RETURN_IF_ERROR(c->GetAttr("max_edges", &max_edges));

      c->set_output(0, c->Vector(max_graphs)); // n_node
      c->set_output(1, c->Vector(max_graphs)); // n_edge
      c->set_output(2, c->Vector(max_edges));  // senders
      c->set_output(3, c->Vector(max_edges));  // receivers
      c->set_output(4, c->Vector(max_edges));  // edge_features
      c->set_output(5, c->Vector(max_nodes));  // node_ids
      c->set_output(6, c->Vector(max_nodes));  // node_labels
      c->set_output(7, c->Vector(max_nodes));  // node_types
      c->set_output(8, c->Vector(max_nodes));  // candidate_mask
      
      return absl::OkStatus();
    });

// Implement the OpKernel
class BuildEnrichedGraphOp : public OpKernel {
 private:
  int max_graphs_;
  int max_nodes_;
  int max_edges_;

 public:
  explicit BuildEnrichedGraphOp(OpKernelConstruction* context) : OpKernel(context) {
    // Read the attributes when the graph is built
    OP_REQUIRES_OK(context, context->GetAttr("max_graphs", &max_graphs_));
    OP_REQUIRES_OK(context, context->GetAttr("max_nodes", &max_nodes_));
    OP_REQUIRES_OK(context, context->GetAttr("max_edges", &max_edges_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the inputs
    const Tensor& user_ids_tensor = context->input(0);
    const Tensor& candidate_ids_tensor = context->input(2);

    // Get dynamic batch size from input (e.g., 1 for inference, 256 for training)
    int batch_size = user_ids_tensor.dim_size(0);
    int num_candidates = candidate_ids_tensor.dim_size(1);

    // Allocate Output Tensors using the attribute sizes
    Tensor* n_node_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({max_graphs_}), &n_node_tensor));
    
    Tensor* node_ids_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(5, TensorShape({max_nodes_}), &node_ids_tensor));

    // TODO: if were continuing this, would add the logic here.
  }
};

REGISTER_KERNEL_BUILDER(Name("BuildEnrichedGraph").Device(DEVICE_CPU), BuildEnrichedGraphOp);
