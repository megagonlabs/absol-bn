"""Typed configuration objects for all public model classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True, kw_only=True)
class BaseModelConfig:
    ignore_columns: Tuple[str, ...] = ()
    random_seed: Optional[int] = None
    verbose: bool = True
    fit_parameters: bool = False
    max_workers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "ignore_columns", tuple(self.ignore_columns))
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")


@dataclass(frozen=True, kw_only=True)
class LLMSettings:
    model_name: str
    provider: str = "openai"
    max_tokens: int = 4096
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    variable_descriptions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        object.__setattr__(self, "variable_descriptions", dict(self.variable_descriptions))


@dataclass(frozen=True, kw_only=True)
class GESSettings:
    score_method: str = "bdeu"
    score_params: Optional[Dict[str, Any]] = None
    lambda_value: Optional[float] = None
    structure_algorithm: str = "ges"

    def __post_init__(self) -> None:
        if self.structure_algorithm not in {"ges", "dges"}:
            raise ValueError("structure_algorithm must be 'ges' or 'dges'")


@dataclass(frozen=True, kw_only=True)
class BaggingConfig(BaseModelConfig):
    num_samples: int
    num_rows_per_sample: int
    num_columns_per_sample: int
    max_parents_per_node: int
    max_nodes_per_markov_blanket: Optional[int] = None
    parent_frequency_cutoff: float = 0.0
    candidate_edges_path: Optional[str] = None
    save_candidate_edges_path: Optional[str] = None
    sample_batch_size: int = 10
    ges: GESSettings = field(default_factory=GESSettings)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "num_samples", "num_rows_per_sample", "num_columns_per_sample",
            "max_parents_per_node", "sample_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.parent_frequency_cutoff <= 1.0:
            raise ValueError("parent_frequency_cutoff must be between 0 and 1")
        if self.max_nodes_per_markov_blanket is not None and self.max_nodes_per_markov_blanket <= 0:
            raise ValueError("max_nodes_per_markov_blanket must be positive when set")


@dataclass(frozen=True, kw_only=True)
class LLMBaggingConfig(BaggingConfig):
    llm: LLMSettings
    parent_ordering: bool = False
    parent_ordering_show_counts: bool = True
    structure_refinement: bool = False
    structure_refinement_show_support: bool = True
    num_refinement_iterations: Optional[int] = None
    column_grouping: bool = False
    column_upweight_factor: float = 5.0
    cycle_arbitration: bool = False
    adaptive_bagging: bool = False
    num_adaptive_passes: int = 1
    num_targeted_samples_per_node: int = 10
    uncertainty_entropy_threshold: float = 0.7
    min_co_occurrence_for_uncertainty: int = 5
    confounder_redirection: bool = False
    orphan_repair: bool = False
    cache_path: Optional[str] = None
    llm_max_workers: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "num_adaptive_passes", "num_targeted_samples_per_node",
            "min_co_occurrence_for_uncertainty", "llm_max_workers",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_refinement_iterations is not None and self.num_refinement_iterations <= 0:
            raise ValueError("num_refinement_iterations must be positive when set")
        if self.column_upweight_factor <= 0:
            raise ValueError("column_upweight_factor must be positive")
        if not 0.0 <= self.uncertainty_entropy_threshold <= 1.0:
            raise ValueError("uncertainty_entropy_threshold must be between 0 and 1")


@dataclass(frozen=True, kw_only=True)
class GESConfig(BaseModelConfig):
    max_parents_per_node: int
    ges: GESSettings = field(default_factory=GESSettings)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_parents_per_node <= 0:
            raise ValueError("max_parents_per_node must be positive")


@dataclass(frozen=True, kw_only=True)
class PCConfig(BaseModelConfig):
    pass


@dataclass(frozen=True, kw_only=True)
class PromptBNConfig(BaseModelConfig):
    llm: LLMSettings


@dataclass(frozen=True, kw_only=True)
class BFSConfig(BaseModelConfig):
    llm: LLMSettings
    include_statistics: bool = False


@dataclass(frozen=True, kw_only=True)
class LLMCDConfig(BaseModelConfig):
    llm: LLMSettings
    alpha: float = 0.05
    ci_test: str = "chisq"
    ci_borderline_threshold: float = 0.001
    max_iterations: int = 1
    max_llm_calls: int = 20000
    max_cycles_per_pass: int = 200
    cache_path: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.ci_borderline_threshold < 0:
            raise ValueError("ci_borderline_threshold must be non-negative")
        for name in ("max_iterations", "max_llm_calls", "max_cycles_per_pass"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
