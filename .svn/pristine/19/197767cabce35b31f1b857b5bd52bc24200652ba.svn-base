/***********************************************************************
    make some changes on flann index.
    
    1. modify the add point function so that when the index was rebuild it
    happen in a new thread ,and can perform simultaneously with search action

*************************************************************************/

#ifndef MULTI_THREAD_INDEX_H_H
#define MULTI_THREAD_INDEX_H_H


#include <vector>
#include <string>
#include <cassert>
#include <cstdio>

#include "flann/general.h"
#include "flann/util/matrix.h"
#include "flann/util/params.h"
#include "flann/util/saving.h"

#include "flann/algorithms/all_indices.h"


#include "MultiThreadHCindex.h"

namespace flann
{
    template<typename Distance>
    class MultiThreadIndex
    {
    public:
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::ResultType DistanceType;
        typedef MultiThreadHCindex<Distance> IndexType;
        MultiThreadIndex()
        {
            
        }
        MultiThreadIndex(const IndexParams& params, Distance distance = Distance())
            : index_params_(params)
        {
            flann_algorithm_t index_type = get_param<flann_algorithm_t>(params, "algorithm");
            loaded_ = false;

            Matrix<ElementType> features;
            if (index_type == FLANN_INDEX_SAVED) {

                std::string filename = get_param<std::string>(params, "filename");
                FILE *fin = fopen(filename.c_str(), "rb");
                if (fin == NULL)
                {
                    nnIndex_ = NULL;
                }
                IndexHeader header = load_header(fin);
                if (header.h.data_type != flann_datatype_value<ElementType>::value) {
                    throw FLANNException("Datatype of saved index is different than of the one to be created.");
                }

                index_params_["algorithm"] = header.h.index_type;

                nnIndex_ = new IndexType(index_params_, distance);
                fclose(fin);

                loaded_ = true;
            }
            else {

                nnIndex_ = new IndexType(params, distance);
            }
        }

        //must set dataset before call this function!
        void load()
        {
            std::string filename = get_param<std::string>(index_params_, "filename");
            FILE *fin = fopen(filename.c_str(), "rb");

            //rewind(fin);
            nnIndex_->loadIndex(fin);
            fclose(fin);

        }

        MultiThreadIndex(const Matrix<ElementType>& features, const IndexParams& params, Distance distance = Distance())
            : index_params_(params)
        {
            flann_algorithm_t index_type = get_param<flann_algorithm_t>(params, "algorithm");
            loaded_ = false;

            if (index_type == FLANN_INDEX_SAVED) {
                nnIndex_ = load_saved_index(features, get_param<std::string>(params, "filename"), distance);
                loaded_ = true;
            }
            else {
                //flann_algorithm_t index_type = get_param<flann_algorithm_t>(params, "algorithm");
                //nnIndex_ = create_index_by_type<Distance>(index_type, features, params, distance);
                nnIndex_ = new IndexType(features, params, distance);
            }
        }


        MultiThreadIndex(const MultiThreadIndex& other) : loaded_(other.loaded_), index_params_(other.index_params_)
        {
            nnIndex_ = other.nnIndex_->clone();
        }
        void extendDataset(const Matrix<ElementType>& new_points)
        {
            nnIndex_->extendDataset(new_points);
        }
        MultiThreadIndex& operator=(MultiThreadIndex other)
        {
            this->swap(other);
            return *this;
        }

        virtual ~MultiThreadIndex()
        {
            delete nnIndex_;
        }

        /**
        * Builds the index.
        */
        void buildIndex()
        {
            if (!loaded_) {
                nnIndex_->buildIndex();
            }
        }

        void buildIndex(const Matrix<ElementType>& points)
        {
            nnIndex_->buildIndex(points);
        }

        void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
        {
            nnIndex_->addPoints(points, rebuild_threshold);
        }

        /**
        * Remove point from the index
        * @param index MultiThreadIndex of point to be removed
        */
        void removePoint(size_t point_id)
        {
            nnIndex_->removePoint(point_id);
        }

        /**
        * Returns pointer to a data point with the specified id.
        * @param point_id the id of point to retrieve
        * @return
        */
        ElementType* getPoint(size_t point_id)
        {
            return nnIndex_->getPoint(point_id);
        }

        /**
        * Save index to file
        * @param filename
        */
        void save(std::string filename)
        {
            FILE* fout = fopen(filename.c_str(), "wb");
            if (fout == NULL) {
                throw FLANNException("Cannot open file");
            }
            nnIndex_->saveIndex(fout);
            fclose(fout);
        }

        /**
        * \returns number of features in this index.
        */
        size_t veclen() const
        {
            return nnIndex_->veclen();
        }

        /**
        * \returns The dimensionality of the features in this index.
        */
        size_t size() const
        {
            return nnIndex_->size();
        }

        /**
        * \returns The index type (kdtree, kmeans,...)
        */
        flann_algorithm_t getType() const
        {
            return nnIndex_->getType();
        }

        /**
        * \returns The amount of memory (in bytes) used by the index.
        */
        int usedMemory() const
        {
            return nnIndex_->usedMemory();
        }


        /**
        * \returns The index parameters
        */
        IndexParams getParameters() const
        {
            return nnIndex_->getParameters();
        }

        /**
        * \brief Perform k-nearest neighbor search
        * \param[in] queries The query points for which to find the nearest neighbors
        * \param[out] indices The indices of the nearest neighbors found
        * \param[out] dists Distances to the nearest neighbors found
        * \param[in] knn Number of nearest neighbors to return
        * \param[in] params Search parameters
        */
        int knnSearch(const Matrix<ElementType>& queries,
                      Matrix<size_t>& indices,
                      Matrix<DistanceType>& dists,
                      size_t knn,
                      const SearchParams& params) const
        {
            return nnIndex_->knnSearch(queries, indices, dists, knn, params);
        }

        /**
        *
        * @param queries
        * @param indices
        * @param dists
        * @param knn
        * @param params
        * @return
        */
        int knnSearch(const Matrix<ElementType>& queries,
                      Matrix<int>& indices,
                      Matrix<DistanceType>& dists,
                      size_t knn,
                      const SearchParams& params) const
        {
            return nnIndex_->knnSearch(queries, indices, dists, knn, params);
        }

        /**
        * \brief Perform k-nearest neighbor search
        * \param[in] queries The query points for which to find the nearest neighbors
        * \param[out] indices The indices of the nearest neighbors found
        * \param[out] dists Distances to the nearest neighbors found
        * \param[in] knn Number of nearest neighbors to return
        * \param[in] params Search parameters
        */
        int knnSearch(const Matrix<ElementType>& queries,
                      std::vector< std::vector<size_t> >& indices,
                      std::vector<std::vector<DistanceType> >& dists,
                      size_t knn,
                      const SearchParams& params)
        {
            return nnIndex_->knnSearch(queries, indices, dists, knn, params);
        }

        /**
        *
        * @param queries
        * @param indices
        * @param dists
        * @param knn
        * @param params
        * @return
        */
        int knnSearch(const Matrix<ElementType>& queries,
                      std::vector< std::vector<int> >& indices,
                      std::vector<std::vector<DistanceType> >& dists,
                      size_t knn,
                      const SearchParams& params) const
        {
            return nnIndex_->knnSearch(queries, indices, dists, knn, params);
        }

        /**
        * \brief Perform radius search
        * \param[in] queries The query points
        * \param[out] indices The indinces of the neighbors found within the given radius
        * \param[out] dists The distances to the nearest neighbors found
        * \param[in] radius The radius used for search
        * \param[in] params Search parameters
        * \returns Number of neighbors found
        */
        int radiusSearch(const Matrix<ElementType>& queries,
                         Matrix<size_t>& indices,
                         Matrix<DistanceType>& dists,
                         float radius,
                         const SearchParams& params) const
        {
            return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
        }

        /**
        *
        * @param queries
        * @param indices
        * @param dists
        * @param radius
        * @param params
        * @return
        */
        int radiusSearch(const Matrix<ElementType>& queries,
                         Matrix<int>& indices,
                         Matrix<DistanceType>& dists,
                         float radius,
                         const SearchParams& params) const
        {
            return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
        }

        /**
        * \brief Perform radius search
        * \param[in] queries The query points
        * \param[out] indices The indinces of the neighbors found within the given radius
        * \param[out] dists The distances to the nearest neighbors found
        * \param[in] radius The radius used for search
        * \param[in] params Search parameters
        * \returns Number of neighbors found
        */
        int radiusSearch(const Matrix<ElementType>& queries,
                         std::vector< std::vector<size_t> >& indices,
                         std::vector<std::vector<DistanceType> >& dists,
                         float radius,
                         const SearchParams& params) const
        {
            return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
        }

        /**
        *
        * @param queries
        * @param indices
        * @param dists
        * @param radius
        * @param params
        * @return
        */
        int radiusSearch(const Matrix<ElementType>& queries,
                         std::vector< std::vector<int> >& indices,
                         std::vector<std::vector<DistanceType> >& dists,
                         float radius,
                         const SearchParams& params) const
        {
            return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
        }

    private:
        IndexType* load_saved_index(const Matrix<ElementType>& dataset, const std::string& filename, Distance distance)
        {
            FILE* fin = fopen(filename.c_str(), "rb");
            if (fin == NULL) {
                return NULL;
            }
            IndexHeader header = load_header(fin);
            if (header.h.data_type != flann_datatype_value<ElementType>::value) {
                throw FLANNException("Datatype of saved index is different than of the one to be created.");
            }

            IndexParams params;
            params["algorithm"] = header.h.index_type;
            //IndexType* nnIndex = create_index_by_type<Distance>(header.index_type, dataset, params, distance);
            IndexType* nnIndex = new IndexType(dataset, params, distance);
            rewind(fin);
            nnIndex->loadIndex(fin);
            fclose(fin);

            return nnIndex;
        }

        void swap(MultiThreadIndex& other)
        {
            std::swap(nnIndex_, other.nnIndex_);
            std::swap(loaded_, other.loaded_);
            std::swap(index_params_, other.index_params_);
        }

    private:
        /** Pointer to actual index class */
        IndexType* nnIndex_;
        /** Indices if the index was loaded from a file */
        bool loaded_;
        /** Parameters passed to the index */
        IndexParams index_params_;
    };
}
#endif