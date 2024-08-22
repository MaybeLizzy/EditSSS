from ..trainer import MEND
from ..trainer import SERAC, SERAC_MULTI


ALG_TRAIN_DICT = {
    'MEND': MEND,
    'MEND_EIGEN': MEND,
    'MEND_EIGEN_NEW': MEND,
    'SERAC': SERAC,
    'SERAC_EIGEN': SERAC,
    'SERAC_MULTI': SERAC_MULTI,
}