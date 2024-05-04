challenges = [
  {
    "category": "Energy measurement",
    "subcategories": [
      {
        "name": "Instrumentation challenges",
        "challenges": [
          {
            "name": "Instrumentation overhead",
            "description": "Instrumented code has additional instructions that may account for overhead and therefore may impact the performance and energy consumption of the code.",
            "mitigation": [],
            "keywords": ["instrumentation", "overhead", "performance", "code", "energy"],
            "query_id": 1,
            "average_similarity": 0.4478
          },
          {
            "name": "Noise in measurement",
            "description": "Background processes running on the machine during energy measurement introduce noise and overheads, affecting the accuracy of measured energy.",
            "mitigation": [],
            "keywords": ["energy", "measurement", "noise", "process", "profiling"],
            "query_id": 2,
            "average_similarity": 0.3844
          }
        ]
      },
      {
        "name": "Hardware variability",
        "challenges": [
          {
            "name": "Hardware configuration",
            "description": "Different hardware configurations can lead to variations in energy consumption values for the same project.",
            "mitigation": [],
            "keywords": ["hardware", "energy", "configuration", "device", "power"],
            "query_id": 3,
            "average_similarity": 0.3357
          },
          {
            "name": "Calibration issues",
            "description": "Energy measurement tools require calibration to account for hardware variations.",
            "mitigation": [],
            "keywords": ["calibration", "energy", "measurement", "hardware", "power"],
            "query_id": 4,
            "average_similarity": 0.3616
          },
          {
            "name": "GPU usage",
            "description": "The selected subject systems must utilize GPU in an optimized manner. Failing to ensure this challenge may introduce incorrect and inconsistent energy data collection.",
            "mitigation": [],
            "keywords": ["GPU", "energy", "optimization", "data", "performance"],
            "query_id": 5,
            "average_similarity": 0.3195
          }
        ]
      },
      {
        "name": "Granularity of energy attribution",
        "challenges": [
          {
            "name": "Precision limits",
            "description": "Existing software tools, due to Intel RAPL limitation, do not permit energy measurements at intervals smaller than 1ms.",
            "mitigation": [],
            "keywords": ["energy", "RAPL", "measurement", "precision", "power"],
            "query_id": 6,
            "average_similarity": 0.3469
          },
          {
            "name": "Precision overhead balance",
            "description": "Observing energy consumption at a high frequency improves the precision of observed data; however, such high frequencies also introduce computation overhead that may introduce noise.",
            "mitigation": [],
            "keywords": ["energy", "precision", "overhead", "frequency", "noise"],
            "query_id": 7,
            "average_similarity": 0.3741
          }
        ]
      }
    ]
  },
  {
    "category": "Patching",
    "subcategories": [
      {
        "name": "Patch generation",
        "challenges": [
          {
            "name": "Correctness of patches",
            "description": "Each identified patch location (in our case, each TensorFlow API) must be correctly patched to record the correct energy consumption of the API and not introduce new syntactic or semantic issues.",
            "mitigation": [],
            "keywords": ["energy", "patch", "TensorFlow", "correctness", "API"],
            "query_id": 8,
            "average_similarity": 0.4110
          },
          {
            "name": "Patch coverage",
            "description": "Each patch location must be identified correctly to avoid missing code that is supposed to be patched and measured.",
            "mitigation": [],
            "keywords": ["patch", "code", "coverage", "identification", "energy"],
            "query_id": 9,
            "average_similarity": 0.3602
          }
        ]
      }
    ]
  },
  {
    "category": "Execution environment",
    "subcategories": [
      {
        "name": "Hardware incompatibility",
        "challenges": [
          {
            "name": "Hardware incompatibility",
            "description": "Compatibility issues arise when using framework versions (e.g. TensorFlow) that are not compatible with the machine's hardware or software dependencies.",
            "mitigation": [],
            "keywords": ["compatibility", "hardware", "version", "TensorFlow", "dependency"],
            "query_id": 10,
            "average_similarity": 0.3068
          }
        ]
      },
      {
        "name": "GPU challenges",
        "challenges": [
          {
            "name": "Memory management",
            "description": "CUDA memory allocation errors may arise when a process cannot allocate sufficient memory during data processing on a GPU.",
            "mitigation": [],
            "keywords": ["memory", "GPU", "CUDA", "allocation", "management"],
            "query_id": 11,
            "average_similarity": 0.4511
          },
          {
            "name": "Container issues",
            "description": "Incompatibility of Docker containers with specific GPUs and TensorFlow versions may hinder the replication of a project.",
            "mitigation": [],
            "keywords": ["Docker", "container", "GPU", "TensorFlow", "version"],
            "query_id": 12,
            "average_similarity": 0.3467
          }
        ]
      }
    ]
  }
]