from dm_control import suite
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control import viewer
# from soccer_body_components import create_soccer_environment
import numpy as np
from asset_components_new import create_flags_and_creatures
import math
import json

# Given JSON input
json_input_pruned_2seg = [
  {
    "UniqueId": 0,
    "TypeId": 1,
    "Position": {"x": 0.0, "y": -5.33999634, "z": 0.0},
    "Rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
    "Size": {"x": 0.7817855, "y": 1.3171674, "z": 0.436341643},
    "ParentUniqueId": None,
    "JointType": None,
    "JointAnchorPos": None,
    "JointAxis": None,
    "Color": {"x": 27.0, "y": 255.0, "z": 233.0}
  },
  {
    "UniqueId": 1,
    "TypeId": 3,
    "Position": {"x": -0.373039246, "y": -4.022829, "z": 0.06939888},
    "Rotation": {"x": 0.0008985509, "y": 0.00109280727, "z": 0.004466458},
    "Size": {"x": 0.595344, "y": 0.419512331, "z": 0.388579845},
    "ParentUniqueId": 0,
    "JointType": "hinge",
    "JointAnchorPos": {"x": -0.477163136, "y": 0.99999994, "z": 0.159047112},
    "JointAxis": {"x": 0.0, "y": 1.0, "z": 0.0},
    "Color": {"x": 187.0, "y": 84.0, "z": 213.0}
  }
]

# TODO: replace nulls with Nones during reading of the json file
json_input_full_2seg = [
  {
    "UniqueId": 0,
    "TypeId": 1,
    "Position": {
      "x": 0.0,
      "y": -5.33999634,
      "z": 0.0,
      "normalized": {
        "x": 0.0,
        "y": -1.0,
        "z": 0.0,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 5.33999634,
      "sqrMagnitude": 28.51556
    },
    "LocalPosition": {
      "x": 0.0,
      "y": -5.33999634,
      "z": 0.0,
      "normalized": {
        "x": 0.0,
        "y": -1.0,
        "z": 0.0,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 5.33999634,
      "sqrMagnitude": 28.51556
    },
    "Rotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "LocalRotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "Size": {
      "x": 0.7817855,
      "y": 1.3171674,
      "z": 0.436341643,
      "normalized": {
        "x": 0.490872949,
        "y": 0.827032268,
        "z": 0.273973256,
        "normalized": {
          "x": 0.490872979,
          "y": 0.8270323,
          "z": 0.2739733,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 1.59264326,
      "sqrMagnitude": 2.53651237
    },
    "ParentUniqueId": None,
    "JointType": None,
    "JointAnchorPos": None,
    "JointAxis": None,
    "Color": {
      "x": 27.0,
      "y": 255.0,
      "z": 233.0,
      "normalized": {
        "x": 0.07792833,
        "y": 0.735989749,
        "z": 0.672492564,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 346.472229,
      "sqrMagnitude": 120043.0
    }
  },
  {
    "UniqueId": 1,
    "TypeId": 3,
    "Position": {
      "x": -0.373039246,
      "y": -4.022829,
      "z": 0.06939888,
      "normalized": {
        "x": -0.0923208147,
        "y": -0.99558115,
        "z": 0.0171750318,
        "normalized": {
          "x": -0.09232082,
          "y": -0.9955812,
          "z": 0.0171750337,
          "magnitude": 1.0,
          "sqrMagnitude": 1.0
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 4.040684,
      "sqrMagnitude": 16.3271275
    },
    "LocalPosition": {
      "x": -0.373039246,
      "y": -4.022829,
      "z": 0.06939888,
      "normalized": {
        "x": -0.0923208147,
        "y": -0.99558115,
        "z": 0.0171750318,
        "normalized": {
          "x": -0.09232082,
          "y": -0.9955812,
          "z": 0.0171750337,
          "magnitude": 1.0,
          "sqrMagnitude": 1.0
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 4.040684,
      "sqrMagnitude": 16.3271275
    },
    "Rotation": {
      "x": 0.0008985509,
      "y": 0.00109280727,
      "z": 0.004466458,
      "normalized": {
        "x": 0.191785961,
        "y": 0.233247876,
        "z": 0.953317165,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.00468517561,
      "sqrMagnitude": 2.19508711E-05
    },
    "LocalRotation": {
      "x": 0.0008985509,
      "y": 0.00109280727,
      "z": 0.004466458,
      "normalized": {
        "x": 0.191785961,
        "y": 0.233247876,
        "z": 0.953317165,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.00468517561,
      "sqrMagnitude": 2.19508711E-05
    },
    "Size": {
      "x": 0.595344,
      "y": 0.419512331,
      "z": 0.388579845,
      "normalized": {
        "x": 0.721208334,
        "y": 0.5082033,
        "z": 0.470731258,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.8254813,
      "sqrMagnitude": 0.6814194
    },
    "ParentUniqueId": 0,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": -0.477163136,
      "y": 0.99999994,
      "z": 0.159047112,
      "normalized": {
        "x": -0.4262798,
        "y": 0.893362761,
        "z": 0.142086774,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 1.119366,
      "sqrMagnitude": 1.25298047
    },
    "JointAxis": {
      "x": 0.0,
      "y": 1.0,
      "z": 0.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 187.0,
      "y": 84.0,
      "z": 213.0,
      "normalized": {
        "x": 0.632558644,
        "y": 0.284143984,
        "z": 0.720508,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 295.624756,
      "sqrMagnitude": 87394.0
    }
  }
]

json_input_full_allseg = [
  {
    "UniqueId": 0,
    "TypeId": 1,
    "Position": {
      "x": 0.0,
      "y": -5.33999634,
      "z": 0.0,
      "normalized": {
        "x": 0.0,
        "y": -1.0,
        "z": 0.0,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 5.33999634,
      "sqrMagnitude": 28.51556
    },
    "LocalPosition": {
      "x": 0.0,
      "y": -5.33999634,
      "z": 0.0,
      "normalized": {
        "x": 0.0,
        "y": -1.0,
        "z": 0.0,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 5.33999634,
      "sqrMagnitude": 28.51556
    },
    "Rotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "LocalRotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "Size": {
      "x": 0.7817855,
      "y": 1.3171674,
      "z": 0.436341643,
      "normalized": {
        "x": 0.490872949,
        "y": 0.827032268,
        "z": 0.273973256,
        "normalized": {
          "x": 0.490872979,
          "y": 0.8270323,
          "z": 0.2739733,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 1.59264326,
      "sqrMagnitude": 2.53651237
    },
    "ParentUniqueId": None,
    "JointType": None,
    "JointAnchorPos": None,
    "JointAxis": None,
    "Color": {
      "x": 27.0,
      "y": 255.0,
      "z": 233.0,
      "normalized": {
        "x": 0.07792833,
        "y": 0.735989749,
        "z": 0.672492564,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 346.472229,
      "sqrMagnitude": 120043.0
    }
  },
  {
    "UniqueId": 1,
    "TypeId": 3,
    "Position": {
      "x": -0.373039246,
      "y": -4.022829,
      "z": 0.06939888,
      "normalized": {
        "x": -0.0923208147,
        "y": -0.99558115,
        "z": 0.0171750318,
        "normalized": {
          "x": -0.09232082,
          "y": -0.9955812,
          "z": 0.0171750337,
          "magnitude": 1.0,
          "sqrMagnitude": 1.0
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 4.040684,
      "sqrMagnitude": 16.3271275
    },
    "LocalPosition": {
      "x": -0.373039246,
      "y": -4.022829,
      "z": 0.06939888,
      "normalized": {
        "x": -0.0923208147,
        "y": -0.99558115,
        "z": 0.0171750318,
        "normalized": {
          "x": -0.09232082,
          "y": -0.9955812,
          "z": 0.0171750337,
          "magnitude": 1.0,
          "sqrMagnitude": 1.0
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 4.040684,
      "sqrMagnitude": 16.3271275
    },
    "Rotation": {
      "x": 0.0008985509,
      "y": 0.00109280727,
      "z": 0.004466458,
      "normalized": {
        "x": 0.191785961,
        "y": 0.233247876,
        "z": 0.953317165,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.00468517561,
      "sqrMagnitude": 2.19508711E-05
    },
    "LocalRotation": {
      "x": 0.0008985509,
      "y": 0.00109280727,
      "z": 0.004466458,
      "normalized": {
        "x": 0.191785961,
        "y": 0.233247876,
        "z": 0.953317165,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.00468517561,
      "sqrMagnitude": 2.19508711E-05
    },
    "Size": {
      "x": 0.595344,
      "y": 0.419512331,
      "z": 0.388579845,
      "normalized": {
        "x": 0.721208334,
        "y": 0.5082033,
        "z": 0.470731258,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.8254813,
      "sqrMagnitude": 0.6814194
    },
    "ParentUniqueId": 0,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": -0.477163136,
      "y": 0.99999994,
      "z": 0.159047112,
      "normalized": {
        "x": -0.4262798,
        "y": 0.893362761,
        "z": 0.142086774,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 1.119366,
      "sqrMagnitude": 1.25298047
    },
    "JointAxis": {
      "x": 0.0,
      "y": 1.0,
      "z": 0.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 187.0,
      "y": 84.0,
      "z": 213.0,
      "normalized": {
        "x": 0.632558644,
        "y": 0.284143984,
        "z": 0.720508,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 295.624756,
      "sqrMagnitude": 87394.0
    }
  },
  {
    "UniqueId": 2,
    "TypeId": 2,
    "Position": {
      "x": 0.390892744,
      "y": -5.1863637,
      "z": -0.06310558,
      "normalized": {
        "x": 0.07515063,
        "y": -0.9970983,
        "z": -0.0121322908,
        "normalized": {
          "x": 0.07515064,
          "y": -0.997098446,
          "z": -0.0121322926,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.9999999,
        "sqrMagnitude": 0.9999998
      },
      "magnitude": 5.20145655,
      "sqrMagnitude": 27.05515
    },
    "LocalPosition": {
      "x": 0.390892744,
      "y": -5.1863637,
      "z": -0.06310558,
      "normalized": {
        "x": 0.07515063,
        "y": -0.9970983,
        "z": -0.0121322908,
        "normalized": {
          "x": 0.07515064,
          "y": -0.997098446,
          "z": -0.0121322926,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.9999999,
        "sqrMagnitude": 0.9999998
      },
      "magnitude": 5.20145655,
      "sqrMagnitude": 27.05515
    },
    "Rotation": {
      "x": 0.000849346863,
      "y": 0.00307855778,
      "z": 0.0009179028,
      "normalized": {
        "x": 0.255606532,
        "y": 0.926476,
        "z": 0.276238084,
        "normalized": {
          "x": 0.255606562,
          "y": 0.926476061,
          "z": 0.2762381,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00332286838,
      "sqrMagnitude": 1.10414539E-05
    },
    "LocalRotation": {
      "x": 0.000849346863,
      "y": 0.00307855778,
      "z": 0.0009179028,
      "normalized": {
        "x": 0.255606532,
        "y": 0.926476,
        "z": 0.276238084,
        "normalized": {
          "x": 0.255606562,
          "y": 0.926476061,
          "z": 0.2762381,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00332286838,
      "sqrMagnitude": 1.10414539E-05
    },
    "Size": {
      "x": 0.165260255,
      "y": 0.5905807,
      "z": 0.482132822,
      "normalized": {
        "x": 0.211846247,
        "y": 0.757062256,
        "z": 0.6180435,
        "normalized": {
          "x": 0.211846277,
          "y": 0.7570624,
          "z": 0.618043542,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.9999999,
        "sqrMagnitude": 0.9999998
      },
      "magnitude": 0.7800953,
      "sqrMagnitude": 0.608548641
    },
    "ParentUniqueId": 0,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": 0.49999997,
      "y": 0.116638668,
      "z": -0.144624248,
      "normalized": {
        "x": 0.9373743,
        "y": 0.2186682,
        "z": -0.2711341,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 0.5334048,
      "sqrMagnitude": 0.284520715
    },
    "JointAxis": {
      "x": 0.0,
      "y": 0.0,
      "z": 1.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 120.0,
      "y": 157.0,
      "z": 121.0,
      "normalized": {
        "x": 0.51788646,
        "y": 0.6775681,
        "z": 0.522202134,
        "normalized": {
          "x": 0.5178865,
          "y": 0.677568138,
          "z": 0.5222022,
          "magnitude": 1.00000012,
          "sqrMagnitude": 1.00000024
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 231.711029,
      "sqrMagnitude": 53690.0
    }
  },
  {
    "UniqueId": 3,
    "TypeId": 2,
    "Position": {
      "x": 0.350706816,
      "y": -4.644242,
      "z": 0.177970886,
      "normalized": {
        "x": 0.07524502,
        "y": -0.9964337,
        "z": 0.0381841,
        "normalized": {
          "x": 0.07524503,
          "y": -0.996433735,
          "z": 0.0381841026,
          "magnitude": 1.0,
          "sqrMagnitude": 1.0
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.9999999
      },
      "magnitude": 4.660864,
      "sqrMagnitude": 21.72365
    },
    "LocalPosition": {
      "x": 0.350706816,
      "y": -4.644242,
      "z": 0.177970886,
      "normalized": {
        "x": 0.07524502,
        "y": -0.9964337,
        "z": 0.0381841,
        "normalized": {
          "x": 0.07524503,
          "y": -0.996433735,
          "z": 0.0381841026,
          "magnitude": 1.0,
          "sqrMagnitude": 1.0
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.9999999
      },
      "magnitude": 4.660864,
      "sqrMagnitude": 21.72365
    },
    "Rotation": {
      "x": 0.000849346863,
      "y": 0.00307855778,
      "z": 0.0009179028,
      "normalized": {
        "x": 0.255606532,
        "y": 0.926476,
        "z": 0.276238084,
        "normalized": {
          "x": 0.255606562,
          "y": 0.926476061,
          "z": 0.2762381,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00332286838,
      "sqrMagnitude": 1.10414539E-05
    },
    "LocalRotation": {
      "x": 0.000849346863,
      "y": 0.00307855778,
      "z": 0.0009179028,
      "normalized": {
        "x": 0.255606532,
        "y": 0.926476,
        "z": 0.276238084,
        "normalized": {
          "x": 0.255606562,
          "y": 0.926476061,
          "z": 0.2762381,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00332286838,
      "sqrMagnitude": 1.10414539E-05
    },
    "Size": {
      "x": 0.145482287,
      "y": 0.5199013,
      "z": 0.424432218,
      "normalized": {
        "x": 0.211846277,
        "y": 0.757062256,
        "z": 0.618043542,
        "normalized": {
          "x": 0.211846292,
          "y": 0.7570623,
          "z": 0.6180436,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.9999999
      },
      "magnitude": 0.686735153,
      "sqrMagnitude": 0.471605152
    },
    "ParentUniqueId": 2,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": -0.243193358,
      "y": 0.9179544,
      "z": 0.4999997,
      "normalized": {
        "x": -0.226603374,
        "y": 0.855334044,
        "z": 0.4658911,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 1.07321155,
      "sqrMagnitude": 1.151783
    },
    "JointAxis": {
      "x": 0.0,
      "y": 0.0,
      "z": 1.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 120.0,
      "y": 157.0,
      "z": 121.0,
      "normalized": {
        "x": 0.51788646,
        "y": 0.6775681,
        "z": 0.522202134,
        "normalized": {
          "x": 0.5178865,
          "y": 0.677568138,
          "z": 0.5222022,
          "magnitude": 1.00000012,
          "sqrMagnitude": 1.00000024
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 231.711029,
      "sqrMagnitude": 53690.0
    }
  },
  {
    "UniqueId": 4,
    "TypeId": 2,
    "Position": {
      "x": 0.1708703,
      "y": -4.022829,
      "z": 0.191222191,
      "normalized": {
        "x": 0.04238912,
        "y": -0.997974336,
        "z": 0.04743797,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 4.03099442,
      "sqrMagnitude": 16.2489147
    },
    "LocalPosition": {
      "x": 0.1708703,
      "y": -4.022829,
      "z": 0.191222191,
      "normalized": {
        "x": 0.04238912,
        "y": -0.997974336,
        "z": 0.04743797,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 4.03099442,
      "sqrMagnitude": 16.2489147
    },
    "Rotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "LocalRotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "Size": {
      "x": 0.386039972,
      "y": 1.37956786,
      "z": 1.126239,
      "normalized": {
        "x": 0.211846262,
        "y": 0.7570623,
        "z": 0.618043542,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 1.82226467,
      "sqrMagnitude": 3.32064867
    },
    "ParentUniqueId": 0,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": 0.218564168,
      "y": 0.99999994,
      "z": 0.4382396,
      "normalized": {
        "x": 0.196290344,
        "y": 0.8980902,
        "z": 0.3935787,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 1.11347389,
      "sqrMagnitude": 1.239824
    },
    "JointAxis": {
      "x": 0.0,
      "y": 0.0,
      "z": -1.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 120.0,
      "y": 157.0,
      "z": 121.0,
      "normalized": {
        "x": 0.51788646,
        "y": 0.6775681,
        "z": 0.522202134,
        "normalized": {
          "x": 0.5178865,
          "y": 0.677568138,
          "z": 0.5222022,
          "magnitude": 1.00000012,
          "sqrMagnitude": 1.00000024
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 231.711029,
      "sqrMagnitude": 53690.0
    }
  },
  {
    "UniqueId": 5,
    "TypeId": 2,
    "Position": {
      "x": 0.2647531,
      "y": -2.75645018,
      "z": 0.7543421,
      "normalized": {
        "x": 0.0922471061,
        "y": -0.960421443,
        "z": 0.2628331,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 2.87004232,
      "sqrMagnitude": 8.23714352
    },
    "LocalPosition": {
      "x": 0.2647531,
      "y": -2.75645018,
      "z": 0.7543421,
      "normalized": {
        "x": 0.0922471061,
        "y": -0.960421443,
        "z": 0.2628331,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 2.87004232,
      "sqrMagnitude": 8.23714352
    },
    "Rotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "LocalRotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "Size": {
      "x": 0.339839548,
      "y": 1.21446419,
      "z": 0.9914531,
      "normalized": {
        "x": 0.211846277,
        "y": 0.7570623,
        "z": 0.618043542,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 1.60418,
      "sqrMagnitude": 2.57339334
    },
    "ParentUniqueId": 4,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": 0.243194491,
      "y": 0.9179532,
      "z": 0.500000358,
      "normalized": {
        "x": 0.226604536,
        "y": 0.8553333,
        "z": 0.4658919,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 1.07321107,
      "sqrMagnitude": 1.151782
    },
    "JointAxis": {
      "x": 0.0,
      "y": 0.0,
      "z": -1.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 120.0,
      "y": 157.0,
      "z": 121.0,
      "normalized": {
        "x": 0.51788646,
        "y": 0.6775681,
        "z": 0.522202134,
        "normalized": {
          "x": 0.5178865,
          "y": 0.677568138,
          "z": 0.5222022,
          "magnitude": 1.00000012,
          "sqrMagnitude": 1.00000024
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 231.711029,
      "sqrMagnitude": 53690.0
    }
  },
  {
    "UniqueId": 6,
    "TypeId": 2,
    "Position": {
      "x": -0.13718462,
      "y": -4.154008,
      "z": -0.21817112,
      "normalized": {
        "x": -0.032961268,
        "y": -0.9980811,
        "z": -0.05241985,
        "normalized": {
          "x": -0.0329612643,
          "y": -0.998080969,
          "z": -0.05241984,
          "normalized": {
            "x": -0.032961268,
            "y": -0.998081,
            "z": -0.052419845,
            "magnitude": 1.0,
            "sqrMagnitude": 1.00000012
          },
          "magnitude": 0.99999994,
          "sqrMagnitude": 0.99999994
        },
        "magnitude": 1.00000012,
        "sqrMagnitude": 1.00000024
      },
      "magnitude": 4.16199446,
      "sqrMagnitude": 17.3221989
    },
    "LocalPosition": {
      "x": -0.13718462,
      "y": -4.154008,
      "z": -0.21817112,
      "normalized": {
        "x": -0.032961268,
        "y": -0.9980811,
        "z": -0.05241985,
        "normalized": {
          "x": -0.0329612643,
          "y": -0.998080969,
          "z": -0.05241984,
          "normalized": {
            "x": -0.032961268,
            "y": -0.998081,
            "z": -0.052419845,
            "magnitude": 1.0,
            "sqrMagnitude": 1.00000012
          },
          "magnitude": 0.99999994,
          "sqrMagnitude": 0.99999994
        },
        "magnitude": 1.00000012,
        "sqrMagnitude": 1.00000024
      },
      "magnitude": 4.16199446,
      "sqrMagnitude": 17.3221989
    },
    "Rotation": {
      "x": 0.0009045518,
      "y": -0.000434916059,
      "z": -0.00434324844,
      "normalized": {
        "x": 0.20291853,
        "y": -0.09756492,
        "z": -0.9743229,
        "normalized": {
          "x": 0.202918544,
          "y": -0.09756493,
          "z": -0.974323,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00445770938,
      "sqrMagnitude": 1.98711732E-05
    },
    "LocalRotation": {
      "x": 0.0009045518,
      "y": -0.000434916059,
      "z": -0.00434324844,
      "normalized": {
        "x": 0.20291853,
        "y": -0.09756492,
        "z": -0.9743229,
        "normalized": {
          "x": 0.202918544,
          "y": -0.09756493,
          "z": -0.974323,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00445770938,
      "sqrMagnitude": 1.98711732E-05
    },
    "Size": {
      "x": 0.4374656,
      "y": 1.5633446,
      "z": 1.27626884,
      "normalized": {
        "x": 0.211846277,
        "y": 0.7570623,
        "z": 0.618043542,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 2.06501436,
      "sqrMagnitude": 4.26428461
    },
    "ParentUniqueId": 0,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": -0.17547603,
      "y": 0.9004083,
      "z": -0.500000656,
      "normalized": {
        "x": -0.1679579,
        "y": 0.861831,
        "z": -0.4785785,
        "normalized": {
          "x": -0.167957917,
          "y": 0.861831069,
          "z": -0.478578538,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 1.044762,
      "sqrMagnitude": 1.0915277
    },
    "JointAxis": {
      "x": 0.0,
      "y": 0.0,
      "z": 1.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 120.0,
      "y": 157.0,
      "z": 121.0,
      "normalized": {
        "x": 0.51788646,
        "y": 0.6775681,
        "z": 0.522202134,
        "normalized": {
          "x": 0.5178865,
          "y": 0.677568138,
          "z": 0.5222022,
          "magnitude": 1.00000012,
          "sqrMagnitude": 1.00000024
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 231.711029,
      "sqrMagnitude": 53690.0
    }
  },
  {
    "UniqueId": 7,
    "TypeId": 2,
    "Position": {
      "x": -0.243469715,
      "y": -2.71893263,
      "z": 0.419984818,
      "normalized": {
        "x": -0.0881520063,
        "y": -0.984432,
        "z": 0.152062058,
        "normalized": {
          "x": -0.08815201,
          "y": -0.984432042,
          "z": 0.152062073,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 2.76193047,
      "sqrMagnitude": 7.62825966
    },
    "LocalPosition": {
      "x": -0.243469715,
      "y": -2.71893263,
      "z": 0.419984818,
      "normalized": {
        "x": -0.0881520063,
        "y": -0.984432,
        "z": 0.152062058,
        "normalized": {
          "x": -0.08815201,
          "y": -0.984432042,
          "z": 0.152062073,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 2.76193047,
      "sqrMagnitude": 7.62825966
    },
    "Rotation": {
      "x": 0.0009045518,
      "y": -0.000434916059,
      "z": -0.00434324844,
      "normalized": {
        "x": 0.20291853,
        "y": -0.09756492,
        "z": -0.9743229,
        "normalized": {
          "x": 0.202918544,
          "y": -0.09756493,
          "z": -0.974323,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00445770938,
      "sqrMagnitude": 1.98711732E-05
    },
    "LocalRotation": {
      "x": 0.0009045518,
      "y": -0.000434916059,
      "z": -0.00434324844,
      "normalized": {
        "x": 0.20291853,
        "y": -0.09756492,
        "z": -0.9743229,
        "normalized": {
          "x": 0.202918544,
          "y": -0.09756493,
          "z": -0.974323,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00445770938,
      "sqrMagnitude": 1.98711732E-05
    },
    "Size": {
      "x": 0.385110676,
      "y": 1.37624693,
      "z": 1.12352777,
      "normalized": {
        "x": 0.211846277,
        "y": 0.7570623,
        "z": 0.618043542,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 1.817878,
      "sqrMagnitude": 3.30468035
    },
    "ParentUniqueId": 6,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": -0.243194059,
      "y": 0.917953253,
      "z": 0.499999762,
      "normalized": {
        "x": -0.226604208,
        "y": 0.8553337,
        "z": 0.4658915,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 1.07321072,
      "sqrMagnitude": 1.15178132
    },
    "JointAxis": {
      "x": 0.0,
      "y": 0.0,
      "z": 1.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 120.0,
      "y": 157.0,
      "z": 121.0,
      "normalized": {
        "x": 0.51788646,
        "y": 0.6775681,
        "z": 0.522202134,
        "normalized": {
          "x": 0.5178865,
          "y": 0.677568138,
          "z": 0.5222022,
          "magnitude": 1.00000012,
          "sqrMagnitude": 1.00000024
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 231.711029,
      "sqrMagnitude": 53690.0
    }
  },
  {
    "UniqueId": 8,
    "TypeId": 3,
    "Position": {
      "x": 0.321707964,
      "y": -5.32608,
      "z": 0.21817112,
      "normalized": {
        "x": 0.06024217,
        "y": -0.9973474,
        "z": 0.0408541374,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 5.34024525,
      "sqrMagnitude": 28.51822
    },
    "LocalPosition": {
      "x": 0.321707964,
      "y": -5.32608,
      "z": 0.21817112,
      "normalized": {
        "x": 0.06024217,
        "y": -0.9973474,
        "z": 0.0408541374,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 5.34024525,
      "sqrMagnitude": 28.51822
    },
    "Rotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "LocalRotation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "magnitude": 0.0,
      "sqrMagnitude": 0.0
    },
    "Size": {
      "x": 0.7438524,
      "y": 0.524159551,
      "z": 0.485510945,
      "normalized": {
        "x": 0.721208334,
        "y": 0.5082033,
        "z": 0.470731229,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 1.03139734,
      "sqrMagnitude": 1.06378043
    },
    "ParentUniqueId": 0,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": 0.41150412,
      "y": 0.01056547,
      "z": 0.500000656,
      "normalized": {
        "x": 0.6353824,
        "y": 0.0163136,
        "z": 0.772025347,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 0.647648,
      "sqrMagnitude": 0.4194479
    },
    "JointAxis": {
      "x": 0.0,
      "y": -1.0,
      "z": 0.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 187.0,
      "y": 84.0,
      "z": 213.0,
      "normalized": {
        "x": 0.632558644,
        "y": 0.284143984,
        "z": 0.720508,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 295.624756,
      "sqrMagnitude": 87394.0
    }
  }
]


def convert_json_to_blueprint(json_input):
    blueprint = {}
    for item in json_input:
        unique_id = str(item["UniqueId"])
        position = (item["Position"]["x"], item["Position"]["z"], item["Position"]["y"])
        rotation = (item["Rotation"]["x"], item["Rotation"]["z"], item["Rotation"]["y"])
        size = (item["Size"]["x"], item["Size"]["z"], item["Size"]["y"])
        color = (item["Color"]["x"], item["Color"]["y"], item["Color"]["z"])

        # Handle parent and subtraction for position if needed
        if item["ParentUniqueId"] is not None and str(item["ParentUniqueId"]) in blueprint:
            parent_position = blueprint[str(item["ParentUniqueId"])]["position"]
            parent_size = blueprint[str(item["ParentUniqueId"])]["size"]
            # Subtract parent position from current position and adjust Z value
            adjusted_position = tuple(np.subtract(position[:2], parent_position[:2])) + (position[2] + parent_size[2] + 0.00000001,)
        else:
            adjusted_position = position

        # if int(unique_id) != 1:
        #     blueprint[unique_id] = {
        #         'position': adjusted_position,
        #         'rotation': rotation,
        #         'size': size,
        #         'parent_unique_id': item["ParentUniqueId"],
        #         'joint_type': item.get("JointType"),
        #         'joint_anchorpos': None if not item.get("JointAnchorPos") else (
        #             item["JointAnchorPos"]["x"], item["JointAnchorPos"]["z"], item["JointAnchorPos"]["y"]
        #         ),
        #         'joint_axis': None if not item.get("JointAxis") else (
        #             item["JointAxis"]["x"], item["JointAxis"]["z"], item["JointAxis"]["y"]
        #         ),
        #         'color': color
        #     }
        # else:
        #     blueprint[1] = {
        #         'position': tuple(np.subtract((0.00, 0.00, -5.34), (-0.37, 0.07, -4.02)))[:2] + (-4.02 + 0.42 + 0.00000001,),
        #         'rotation': rotation,
        #         'size': size,
        #         'parent_unique_id': item["ParentUniqueId"],
        #         'joint_type': item.get("JointType"),
        #         'joint_anchorpos': None if not item.get("JointAnchorPos") else (
        #             item["JointAnchorPos"]["x"], item["JointAnchorPos"]["z"], item["JointAnchorPos"]["y"]
        #         ),
        #         'joint_axis': None if not item.get("JointAxis") else (
        #             item["JointAxis"]["x"], item["JointAxis"]["z"], item["JointAxis"]["y"]
        #         ),
        #         'color': color
        #     }
            
        blueprint[unique_id] = {
            'position': adjusted_position,
            'rotation': rotation,
            'size': size,
            'parent_unique_id': item["ParentUniqueId"],
            'joint_type': item.get("JointType"),
            'joint_anchorpos': None if not item.get("JointAnchorPos") else (
                item["JointAnchorPos"]["x"], item["JointAnchorPos"]["z"], item["JointAnchorPos"]["y"]
            ),
            'joint_axis': None if not item.get("JointAxis") else (
                item["JointAxis"]["x"], item["JointAxis"]["z"], item["JointAxis"]["y"]
            ),
            'color': color
        }

    return blueprint

blueprint = convert_json_to_blueprint(json_input_full_allseg)

manual_blueprint = {
    '0': {
        # TODO: I think make this (0 0 0), handle shifts elsewhere outside.
        'position': (0.00, 0.00, -5.34), #yz swapped, 
        'rotation': (0.0, 0.0, 0.0), #yz swapped
        'size': (0.78, 0.44, 1.32), #yz swapped
        'parent_unique_id': None,
        'joint_type': None, 
        'joint_anchorpos': None, #yz swapped
        'joint_axis': None, #yz swapped
        'color': (27.00, 255.00, 233.00)
    },
    '1': {
        # TODO: think about direction of the offset (probably will depend on something). but for now for this test case,
        'position': tuple(np.subtract((0.00, 0.00, -5.34), (-0.37, 0.07, -4.02)))[:2] + (-4.02 + 0.42 + 0.00000001,), #yz swapped, then subtracted from the position of the parent segment, then z value replaced accordingly. TODO: verify that this works.. intuitively it's a bit sus
        'rotation': (0.0, 0.0, 0.0), #yz swapped
        'size': (0.60, 0.39, 0.42), #yz swapped
        'parent_unique_id': 1,
        'joint_type': 'hinge',
        'joint_anchorpos': (-0.48, 0.16, 1.00), #yz swapped
        'joint_axis': (0, 0, 1), #yz swapped
        'color': (187.00, 84.00, 213.00)
    }
}


class CustomSoccerEnv(base.Task):
    def __init__(self, xml_string):
        self.xml_string = xml_string
        super().__init__()

    def initialize_episode(self, physics):
        # This method is called at the start of each episode, you can reset the environment here
        pass

    def get_observation(self, physics):
        # Here, return an observation based on the current state of the physics
        return {}

    def get_reward(self, physics):
        # Define and return a reward based on the current state of the environment
        return 0

def load_and_render_soccer_env(xml_string):
    # Parse the XML string to a MuJoCo model
    model = mujoco.wrapper.MjModel.from_xml_string(xml_string)
    physics = mujoco.Physics.from_model(model)
    
    # Create an instance of the environment
    task = CustomSoccerEnv(xml_string)
    env = control.Environment(physics, task, time_limit=20)
    
    # Initialize a step counter
    step_counter = 0

    # Define a dummy policy that does nothing (for demonstration purposes)
    def policy(time_step):
        nonlocal step_counter
        action_spec = env.action_spec()
        # return np.zeros(action_spec.shape)
        # return 0.1 * np.ones(action_spec.shape) 
        action = -200 * math.sin(0.005 * step_counter) * np.ones(action_spec.shape) 
        # Increment the step counter
        step_counter += 1
        return action
    

    
    # Use the dm_control viewer to render the environment
    viewer.launch(env, policy=policy)

# Generate the XML for the soccer environment
# xml_soccer = create_soccer_environment()
# xml_soccer =  create_ant_model(num_creatures=9)
xml_soccer, _ =  create_flags_and_creatures(num_creatures=1, blueprint=blueprint)
print(xml_soccer)

# Load and render the environment
load_and_render_soccer_env(xml_soccer)
