#ifndef NEURO_CAR_EVOLUTION_HPP
#define NEURO_CAR_EVOLUTION_HPP

#include <cstddef>
#include <vector>

#include <dna.hpp>


namespace NeuroCar {

template <typename T>
using Individuals = std::vector<T*>;

template <typename DNAType>
using Population = std::vector<DNAType>;

// TODO:
// How to handle the T entities created at each generation?
template <typename DNAType>
Population<DNAType> evolution(Population<DNAType> const & population);


template <typename DNAType, typename T>
Population<DNAType> simulate(Individuals<T> const & individuals, std::size_t ngenerations);

template <typename DNAType>
void evaluate(Population<DNAType> const & population);

}

#include "evolution.inl"

#endif //NEURO_CAR_EVOLUTION_HPP
