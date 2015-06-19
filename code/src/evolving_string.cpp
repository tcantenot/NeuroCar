#include <evolving_string.hpp>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <random>
#include <string>


#include <iostream>



namespace NeuroCar {

namespace {

static const char ALPHABET[] =
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 ";

}

EvolvingStringDNA::EvolvingStringDNA(EvolvingString * subject):
    DNA(subject),
    m_fitness(0.0)
{

}

void EvolvingStringDNA::randomize()
{
    assert(m_subject);
    m_subject->setGenes(EvolvingStringDNA::RandomString(m_subject->getTarget().size()));
}

EvolvingStringDNA::Fitness EvolvingStringDNA::computeFitness()
{
    assert(m_subject);

    Fitness fitness = 0.0;
    std::string const & genes = m_subject->getGenes();
    std::string const & target = m_subject->getTarget();

    assert(genes.size() == target.size());

    for(auto i = 0u; i < genes.size(); ++i)
    {
        if(genes[i] == target[i]) ++fitness;
    }

    fitness /= static_cast<Fitness>(genes.size());

    m_fitness = fitness;

    return fitness;
}

EvolvingStringDNA::Fitness EvolvingStringDNA::getFitness() const
{
    return m_fitness;
}

EvolvingString * EvolvingStringDNA::crossover(EvolvingStringDNA const & partner) const
{
    assert(m_subject);

    EvolvingString * child = new EvolvingString(*m_subject);

    std::size_t length = m_subject->getGenes().size();

    static std::random_device rd;
    static std::default_random_engine rng(rd());
    std::uniform_int_distribution<> dist(0, length-1);

    uint32_t midpoint = dist(rng);

    auto const & parentAGenes = m_subject->getGenes();
    auto const & parentBGenes = partner.getSubject()->getGenes();
    auto & childGenes = child->getGenes();

    for(auto i = 0u; i < length; ++i)
    {
        // FIXME: does not seem really good crossover function...
        childGenes[i] = i > midpoint ? parentAGenes[i] : parentBGenes[i];
    }

    return child;
}

void EvolvingStringDNA::mutate(MutationRate mutationRate)
{
    static std::random_device rd;
    static std::default_random_engine rng(rd());
    std::uniform_real_distribution<MutationRate> random(0.0, 1.0);

    auto & genes = m_subject->getGenes();
    std::size_t length = genes.size();

    for(auto i = 0u; i < length; ++i)
    {
        if(random(rng) < mutationRate)
        {
            genes[i] = RandomChar();
        }
    }
}


char EvolvingStringDNA::RandomChar()
{
    static std::random_device rd;
    static std::default_random_engine rng(rd());
    static std::uniform_int_distribution<> dist(0, sizeof(ALPHABET)/sizeof(*ALPHABET)-2);

    return ALPHABET[dist(rng)];
}

std::string EvolvingStringDNA::RandomString(std::size_t length)
{
    std::string str;
    str.reserve(length);
    std::generate_n(std::back_inserter(str), length, [&]() { return RandomChar(); });

    return str;
}

void stringEvolution()
{
    Population<EvolvingString> strings;
    for(auto i = 0; i < 200; ++i)
    {
        strings.push_back(new EvolvingString("To be or not to be"));
    }

    EvolutionParams params;
    params.mutationRate = 0.02;

    DNAs<EvolvingStringDNA> dnas = evolve<EvolvingStringDNA>(strings, 10000, params);

    for(auto & dna: dnas)
    {
        auto subject = dna.getSubject();
        std::cout << subject->getGenes() << std::endl;
    }
}

}
