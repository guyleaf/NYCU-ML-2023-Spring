#include <cstddef>
#include <stdexcept>

namespace common
{
    inline int castToInt(unsigned char *bytes, std::size_t count, bool isBigEndian = false)
    {
        if (count > 4)
        {
            throw std::range_error("The size of the bytes array should be smaller than 4.");
        }

        int value = 0;

        for (std::size_t i = 0; i < count; i++)
        {
            if (isBigEndian)
            {
                value |= (bytes[i] << (count - i - 1) * 8);
            }
            else
            {
                value |= (bytes[i] << i * 8);
            }
        }

        return value;
    }
}