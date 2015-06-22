#ifndef NEURO_CAR_EVOLUTION_HPP
#define NEURO_CAR_EVOLUTION_HPP

#include <cstddef>
#include <vector>

#include <dna.hpp>


namespace NeuroCar {

template <typename T>
using Population = std::vector<Individual<T>>;

template <typename DNAType>
using DNAs = std::vector<DNAType>;

struct EvolutionParams
{
    using MutationRate = double;
    using Elitism = uint32_t;

    MutationRate mutationRate = 0.01;
    Elitism elitism = 1;
};

template <typename DNAType, typename T>
DNAs<DNAType> evolve(
    Population<T> const & population,
    std::size_t ngenerations,
    EvolutionParams const & params = { }
);

}

#include "evolution.inl"

#endif //NEURO_CAR_EVOLUTION_HPP
