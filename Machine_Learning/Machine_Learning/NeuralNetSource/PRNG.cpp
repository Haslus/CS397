
#include "PRNG.h"

#include <cstdlib>

namespace PRNG
{

double RandomDouble01()
{
	return static_cast<double>(std::rand()) / RAND_MAX;
}

double RandomNormalizedDouble()
{
    return RandomDouble01() * 2.0 - 1.0;
}

} // namespace PRNG
