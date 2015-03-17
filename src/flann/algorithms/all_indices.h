/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/


#ifndef FLANN_ALL_INDICES_H_
#define FLANN_ALL_INDICES_H_

#include "../general.h"
#include "nn_index.h"

#include "util/dynamic_bitset.h"
//#include "flann/algorithms/linear_index.h"
#include "hierarchical_clustering_index.h"



namespace flann
{

    /**
     * enable_if sfinae helper
     */
    template<bool, typename T = void> struct enable_if{};
    template<typename T> struct enable_if<true, T> { typedef T type; };

    /**
     * disable_if sfinae helper
     */
    template<bool, typename T> struct disable_if{ typedef T type; };
    template<typename T> struct disable_if<true, T> { };

    /**
     * Check if two type are the same
     */
    template <typename T, typename U>
    struct same_type
    {
        enum { value = false };
    };

    template<typename T>
    struct same_type<T, T>
    {
        enum { value = true };
    };


    struct DummyDistance
    {
        typedef float ElementType;
        typedef float ResultType;

        template <typename Iterator1, typename Iterator2>
        ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType /*worst_dist*/ = -1) const
        {
            return ResultType(0);
        }

        template <typename U, typename V>
        inline ResultType accum_dist(const U& a, const V& b, int) const
        {
            return ResultType(0);
        }
    };

    /**
     * Checks if an index and a distance can be used together
     */
    template<template <typename> class MultiThreadIndex, typename Distance, typename ElemType>
    struct valid_combination
    {
        static const bool value = true;

    };


    /*********************************************************
     * Create index
     **********************************************************/

    template <template<typename> class IndexType, typename Distance, typename T>
    inline NNIndex<Distance>* create_index_(const flann::IndexParams& params, const Distance& distance,
                                            typename enable_if<valid_combination<IndexType, Distance, T>::value, void>::type* = 0)
    {
        return new IndexType<Distance>( params, distance);
    }

    //template <template<typename> class IndexType, typename Distance, typename T>
//     inline NNIndex<Distance>* create_index_(flann::Matrix<T> data, const flann::IndexParams& params, const Distance& distance,
//                                             typename disable_if<valid_combination<IndexType, Distance, T>::value, void>::type* = 0)
//     {
//         return NULL;
//     }

    template<typename Distance>
    inline NNIndex<Distance>* create_index_by_type(const flann_algorithm_t index_type, 
                                                   const IndexParams& params, const Distance& distance)
    {
            typedef typename Distance::ElementType ElementType;

            NNIndex<Distance>* nnIndex;

            switch (index_type)
            {
                case FLANN_INDEX_MULTITHREAD:
                    nnIndex = create_index_<MultiThreadHierarchicalIndex, Distance, ElementType>( params, distance);
                    break;
                default:
                    throw FLANNException("Unknown index type");
            }

            if (nnIndex == NULL)
            {
                throw FLANNException("Unsupported index/distance combination");
            }
            return nnIndex;
    }

}

#endif /* FLANN_ALL_INDICES_H_ */
