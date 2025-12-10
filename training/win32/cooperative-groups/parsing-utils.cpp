#include "parsing-utils.h"

// std library
#include <iomanip>
#include <vector>
#include <cmath>
#include <sstream>

namespace dmt {
    size_t prettyPrintFloatMaxWidth(float const* data, size_t count)
    {
        size_t max_width = 0;
        for (size_t i = 0; i < count; ++i)
        {
            float const        v = data[i];
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << v;
            max_width = std::max<size_t>(max_width, oss.str().size());
        }
        return max_width;
    }
} // namespace dmt