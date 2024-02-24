from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
import json


@dataclass
class PeftArguments:
    """
    Arguments pertaining to PEFT methods.
    """

    adapter_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name to use for the adapter. If not specified, the adapter will be named `default`."
            )
        },
    )

    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRa"}
    )
    rank: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Lora layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str."
        },
    )
    rank_pattern: Optional[str] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[str] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )




    # Prompt tuning arguments
    use_prompt_tuning: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt tuning"}
    )
    num_virtual_tokens: int = field(
        default=None,
        metadata={"help": "Number of virtual tokens to use for prompt tuning."}
    )
    virtual_tokens_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Initialize the virtual tokens with the given text. Otherwise, the virtual tokens will be initialized randomly."
            )
        },
    )

    def __post_init__(self):

        # only one of the two can be used
        if self.use_lora and self.use_prompt_tuning:
            raise ValueError("Only one of `use_lora` and `use_prompt_tuning` can be used.")

        # handle lists of str. split the string by comma
        if isinstance(self.target_modules, str):
            self.target_modules = self.target_modules.split(",")
        if isinstance(self.modules_to_save, str):
            self.modules_to_save = self.modules_to_save.split(",")
        if isinstance(self.layers_to_transform, str):
            self.layers_to_transform = self.layers_to_transform.split(",")
        if isinstance(self.layers_pattern, str):
            self.layers_pattern = self.layers_pattern.split(",")
        
        if isinstance(self.rank_pattern, str):
            self.rank_pattern = json.loads(self.rank_pattern)
        if isinstance(self.alpha_pattern, str):
            self.alpha_pattern = json.loads(self.alpha_pattern)
