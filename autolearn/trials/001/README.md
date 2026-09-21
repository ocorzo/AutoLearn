# AutoLearn trial 001

This isolated trial validates whether a single explicitly authorized Mem0 global memory can be internalized by a LoRA candidate. It must not change the active Chaak adapter or production service.

The source fact is recorded in `lesson.json`. Training data, held-out evaluation data, model configuration, candidate adapter, and metrics are created only in later steps. Evaluation must run without Mem0 retrieval or Chatbox conversation context.
