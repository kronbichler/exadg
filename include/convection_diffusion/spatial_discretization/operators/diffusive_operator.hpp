#include "diffusive_operator.h"
#include "../../../../applications/macros/constants.h"

namespace ConvDiff
{

#if DIM_2 && DEGREE_1
template class DiffusiveOperator<2, 1, float>;
template class DiffusiveOperator<2, 1, double>;
#endif

#if DIM_2 && DEGREE_2
template class DiffusiveOperator<2, 2, float>;
template class DiffusiveOperator<2, 2, double>;
#endif

#if DIM_2 && DEGREE_3
template class DiffusiveOperator<2, 3, float>;
template class DiffusiveOperator<2, 3, double>;
#endif

#if DIM_2 && DEGREE_4
template class DiffusiveOperator<2, 4, float>;
template class DiffusiveOperator<2, 4, double>;
#endif

#if DIM_2 && DEGREE_5
template class DiffusiveOperator<2, 5, float>;
template class DiffusiveOperator<2, 5, double>;
#endif

#if DIM_2 && DEGREE_6
template class DiffusiveOperator<2, 6, float>;
template class DiffusiveOperator<2, 6, double>;
#endif

#if DIM_2 && DEGREE_7
template class DiffusiveOperator<2, 7, float>;
template class DiffusiveOperator<2, 7, double>;
#endif

#if DIM_2 && DEGREE_8
template class DiffusiveOperator<2, 8, float>;
template class DiffusiveOperator<2, 8, double>;
#endif

#if DIM_2 && DEGREE_9
template class DiffusiveOperator<2, 9, float>;
template class DiffusiveOperator<2, 9, double>;
#endif

}

