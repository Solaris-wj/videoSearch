
#ifndef INDEX_SORT_H_H
#define INDEX_SORT_H_H
#include <vector>

template<class T, class Comp>
static size_t partition(std::vector<T> &vec, size_t it_low, size_t it_high, std::vector<int> &index, Comp comp)
{
    auto pivot = vec[it_low];
    auto pivot_index = index[it_low];

    while (it_low < it_high)
    {
        while (it_low < it_high  && comp(pivot, vec[it_high]))
            it_high--;
        vec[it_low] = vec[it_high];
        index[it_low] = index[it_high];

        while (it_low < it_high  && !comp(pivot, vec[it_low]))
            it_low++;
        vec[it_high] = vec[it_low];
        index[it_low] = index[it_high];
    }
    vec[it_low] = pivot;
    index[it_low] = pivot_index;
    return it_low;
}

template<class T, class Comp >
void index_sort(std::vector<T> &vec, size_t it_low, size_t it_high, std::vector<int> &index, Comp comp)
{
    if (it_low >= it_high)
    {
        //index[it_low] = it_low;
        return;
    }
    if (index.size() != vec.size())
    {
        index.clear();
        index.resize(vec.size());
        iota(index.begin(), index.end(), 0);
    }
    size_t mid = partition(vec, it_low, it_high, index, comp);

    if (mid > 0)
        index_sort(vec, it_low, mid - 1, index, comp);
    if (mid < it_high)
        index_sort(vec, mid + 1, it_high, index, comp);
}


template<class T> inline
void index_sort(std::vector<T> &vec, size_t it_low, size_t it_high, std::vector<int> &index)
{
    index_sort(vec, it_low, it_high, index, std::less<T>());
}

#endif