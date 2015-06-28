#ifndef NEURO_CAR_EVOLUTION_HPP
#define NEURO_CAR_EVOLUTION_HPP

#include <cstddef>
#include <functional>
#include <vector>

#include <dna.hpp>


namespace NeuroCar {

template <typename T>
using Population = std::vector<Individual<T>>;

template <typename DNAType>
using DNAs = std::vector<DNAType>;


template <typename DNAType>
struct EvolutionParams
{
    using MutationRate = double;
    using Elitism = uint32_t;
    using GenerationHook = std::function<void (std::size_t, DNAs<DNAType> const &)>;

    MutationRate mutationRate = 0.01;
    Elitism elitism = 1;
    GenerationHook preGenHook  = GenerationHook(defaultPreGenHook);
    GenerationHook postGenHook = GenerationHook(defaultPostGenHook);

    private:
        static void defaultPreGenHook(std::size_t, DNAs<DNAType> const &) { }
        static void defaultPostGenHook(std::size_t, DNAs<DNAType> const &) { }
};

template <typename DNAType, typename T>
DNAs<DNAType> evolve(
    Population<T> const & population,
    std::size_t ngenerations,
    EvolutionParams<DNAType> const & params = { }
);

}

#include "evolution.inl"

#endif //NEURO_CAR_EVOLUTION_HPP
