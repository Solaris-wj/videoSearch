#ifndef MULTI_THREAD_HC_INDEX_H_H
#define MULTI_THREAD_HC_INDEX_H_H
#include <flann/algorithms/hierarchical_clustering_index.h>

#include <thread>
#include <functional>
#include <mutex>
#include <memory>
#include <flann/flann.hpp>
namespace flann
{
    template <typename Distance>
    class MultiThreadHCindex : public virtual HierarchicalClusteringIndex<Distance>
    {
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::ResultType DistanceType;
        typedef HierarchicalClusteringIndex<Distance> BaseClass;
    private:
        std::mutex mutex_;
        std::thread th;
    public:
        MultiThreadHCindex(const IndexParams& index_params = HierarchicalClusteringIndexParams(), Distance d = Distance())
            :HierarchicalClusteringIndex(index_params, d)
        {}
        MultiThreadHCindex(const MultiThreadHCindex &other) :HierarchicalClusteringIndex(other)
        {}
        MultiThreadHCindex(const Matrix<ElementType>& inputData, const IndexParams& index_params = HierarchicalClusteringIndexParams(),
                           Distance d = Distance())
                           :HierarchicalClusteringIndex(inputData, index_params, d)
        {}
        ~MultiThreadHCindex()
        {
            if (th.joinable())
                th.join();
        }
        std::mutex& getMutex()
        {
            return mutex_;
        }
        void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2) override
        {
            std::lock_guard<std::mutex> lg(mutex_);
            //assert(points.cols == veclen_);
            veclen_ = points.cols;
            size_t old_size = size_;
                     

            if (rebuild_threshold > 1 && (old_size + points.rows)*rebuild_threshold > old_size)
            {
                extendDataset(points);
                rebuildInThread();
            }
            else 
            {
                for (size_t i = 0; i < points.rows; ++i) 
                {
                    for (int j = 0; j < trees_; j++) 
                    {
                        addPointToTree(tree_roots_[j], old_size + i);
                    }
                }
                //BaseClass::addPoints(points, rebuild_threshold);
            }
        }
        void extendDataset(const Matrix<ElementType>& new_points)
        {
            BaseClass::extendDataset(new_points);
        }
        void rebuildInThread()
        {
            std::shared_ptr<MultiThreadHCindex> temp(new MultiThreadHCindex(*this));
            th=std::thread(&MultiThreadHCindex::buildInThread, this,temp);
            //buildIndex();
        }
        void buildInThread(std::shared_ptr<MultiThreadHCindex> other)
        {
            other->buildIndex();
            std::lock_guard<std::mutex> lg(this->getMutex());
            other->swap(*this);
        }
    };
}

#endif


