#ifndef NON_COPYABLE_H_H
#define NON_COPYABLE_H_H

namespace vs
{
    class NonCopyable
    {
    private:
        NonCopyable() = default;
        NonCopyable(const NonCopyable &) = delete;
        NonCopyable & operator = (const NonCopyable) = delete;
    };
}
#endif
