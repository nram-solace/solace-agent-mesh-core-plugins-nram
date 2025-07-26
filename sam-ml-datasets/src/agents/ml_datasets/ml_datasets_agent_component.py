"""ML Datasets agent component for providing various ML datasets."""

import copy
from typing import Dict, Any

from solace_ai_connector.common.log import log
from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)

from .services.dataset_service import DatasetService
from .actions.get_dataset import GetDataset
from .actions.list_datasets import ListDatasets

# Import version
try:
    from ... import __version__
except ImportError:
    __version__ = "0.1.0+local.unknown"  # Fallback version


info = copy.deepcopy(agent_info)
info.update(
    {
        "agent_name": "ml_datasets",
        "class_name": "MLDatasetsAgentComponent",
        "description": "ML Datasets agent for providing sklearn, seaborn, and synthetic datasets for machine learning workloads",
        "config_parameters": [
            {
                "name": "agent_name",
                "required": True,
                "description": "Name of this ML datasets agent instance",
                "type": "string",
            },
            {
                "name": "default_max_records",
                "required": False,
                "description": "Default maximum number of records to return from datasets",
                "type": "integer",
                "default": 100,
            },
            {
                "name": "enable_sklearn_datasets",
                "required": False,
                "description": "Enable sklearn dataset collection",
                "type": "boolean",
                "default": True,
            },
            {
                "name": "enable_seaborn_datasets",
                "required": False,
                "description": "Enable seaborn dataset collection",
                "type": "boolean",
                "default": True,
            },
            {
                "name": "enable_synthetic_datasets",
                "required": False,
                "description": "Enable synthetic dataset generation",
                "type": "boolean",
                "default": True,
            },
        ],
    }
)


class MLDatasetsAgentComponent(BaseAgentComponent):
    """Component for providing various ML datasets."""

    info = info
    actions = [GetDataset, ListDatasets]

    def __init__(self, module_info: Dict[str, Any] = None, **kwargs):
        """Initialize the ML Datasets agent component.

        Args:
            module_info: Optional module configuration.
            **kwargs: Additional keyword arguments.
        """
        module_info = module_info or info
        
        log.info("ml-datasets: Initializing ML Datasets agent component")
        log.debug("ml-datasets: Available kwargs keys: %s", list(kwargs.keys()))
        
        # Log component_config if present
        if 'component_config' in kwargs:
            log.debug("ml-datasets: component_config keys: %s", list(kwargs['component_config'].keys()))
        
        super().__init__(module_info, **kwargs)

        self.agent_name = self.get_config("agent_name")
        self.default_max_records = self.get_config("default_max_records", 100)
        self.enable_sklearn = self.get_config("enable_sklearn_datasets", True)
        self.enable_seaborn = self.get_config("enable_seaborn_datasets", True)
        self.enable_synthetic = self.get_config("enable_synthetic_datasets", True)

        self.action_list.fix_scopes("<agent_name>", self.agent_name)
        module_info["agent_name"] = self.agent_name

        # Initialize dataset service
        self.dataset_service = DatasetService(default_max_records=self.default_max_records)
        log.info("ml-datasets: Dataset service initialized with max_records=%d", self.default_max_records)

        # Generate and store the agent description
        self._generate_agent_description()

        # Log prominent startup message
        log.info("=" * 80)
        log.info("ml-datasets: 📊  ML DATASETS AGENT (v%s) STARTED SUCCESSFULLY", __version__)
        log.info("=" * 80)
        log.info("ml-datasets: Agent Name: %s", self.agent_name)
        log.info("ml-datasets: Default Max Records: %d", self.default_max_records)
        log.info("ml-datasets: Available Actions: %s", [action.__name__ for action in self.actions])
        log.info("ml-datasets: Sklearn Datasets: %s", "Enabled" if self.enable_sklearn else "Disabled")
        log.info("ml-datasets: Seaborn Datasets: %s", "Enabled" if self.enable_seaborn else "Disabled")
        log.info("ml-datasets: Synthetic Datasets: %s", "Enabled" if self.enable_synthetic else "Disabled")
        
        # Show available datasets summary
        try:
            available = self.dataset_service.list_available_datasets()
            total_datasets = sum(len(datasets) for datasets in available.values())
            log.info("ml-datasets: Total Available Datasets: %d", total_datasets)
            for dtype, datasets in available.items():
                log.info("ml-datasets: - %s: %d datasets", dtype.upper(), len(datasets))
        except Exception as e:
            log.warning("ml-datasets: Could not load dataset summary: %s", str(e))
        
        log.info("=" * 80)
        log.info("ml-datasets: ✅ ML Datasets Agent is ready to provide datasets!")
        log.info("ml-datasets: 🔍 Agent should be available in SAM as 'ml_datasets'")
        log.info("=" * 80)
        
        # Also print to stdout for immediate visibility
        print("=" * 80)
        print(f"📊  ML DATASETS AGENT (v{__version__}) STARTED SUCCESSFULLY")
        print("=" * 80)
        print(f"Agent Name: {self.agent_name}")
        print(f"Default Max Records: {self.default_max_records}")
        print(f"Available Actions: {[action.__name__ for action in self.actions]}")
        print("Dataset Collections:")
        print(f"  - Sklearn: {'Enabled' if self.enable_sklearn else 'Disabled'}")
        print(f"  - Seaborn: {'Enabled' if self.enable_seaborn else 'Disabled'}")  
        print(f"  - Synthetic: {'Enabled' if self.enable_synthetic else 'Disabled'}")
        print("=" * 80)
        print("✅ ML Datasets Agent is ready to provide datasets!")
        print("🔍 Agent should be available in SAM as 'ml_datasets'")
        print("=" * 80)

    def _generate_agent_description(self):
        """Generate and store the agent description."""
        description = "This agent provides access to various ML datasets for training and analysis.\n\n"
        
        description += "**Available Dataset Collections:**\n"
        if self.enable_sklearn:
            description += "- **Sklearn Datasets**: Classic ML datasets (iris, wine, breast_cancer, digits, diabetes, etc.)\n"
        if self.enable_seaborn:
            description += "- **Seaborn Datasets**: Real-world datasets for statistical analysis (tips, flights, titanic, etc.)\n"
        if self.enable_synthetic:
            description += "- **Synthetic Datasets**: Generated datasets for testing (classification, regression, clustering, etc.)\n"
        
        description += f"\n**Default Configuration:**\n"
        description += f"- Maximum records per dataset: {self.default_max_records}\n"
        description += f"- Output formats: JSON, YAML, CSV\n"
        description += f"- Includes metadata and dataset descriptions\n"
        
        description += f"\n**Integration:**\n"
        description += f"- Datasets are compatible with sam-ml-pandas for analysis and EDA\n"
        description += f"- All datasets limited to {self.default_max_records} records by default for efficiency\n"

        self._agent_description = {
            "agent_name": self.agent_name,
            "description": description.strip(),
            "always_open": self.info.get("always_open", False),
            "actions": self.get_actions_summary(),
        }

    def get_agent_summary(self):
        """Get a summary of the agent's capabilities."""
        return self._agent_description