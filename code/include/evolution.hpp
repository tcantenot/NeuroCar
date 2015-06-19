#ifndef NEURO_CAR_EVOLUTION_HPP
#define NEURO_CAR_EVOLUTION_HPP

#include <cstddef>
#include <vector>

#include <dna.hpp>

#include <mpi.hpp>


namespace NeuroCar {

template <typename T>
using Population = std::vector<T*>;

template <typename DNAType>
using DNAs = std::vector<DNAType>;

#if 1
struct EvolutionParams
{
    using MutationRate = double;

    MutationRate mutationRate = 0.01;
    proc_info_t procInfo = { };
};
#endif

template <typename DNAType, typename T>
DNAs<DNAType> evolve(
    Population<T> const & population,
    std::size_t ngenerations,
    EvolutionParams params = { }
);

}

#include "evolution.inl"

#endif //NEURO_CAR_EVOLUTION_HPP
