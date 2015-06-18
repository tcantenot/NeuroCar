#include <evolving_string.hpp>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <random>
#include <string>


#include <iostream>



namespace NeuroCar {

EvolvingStringDNA::EvolvingStringDNA(EvolvingString * subject):
    DNA(subject)
{

}

void EvolvingStringDNA::randomize()
{
    assert(m_subject);
    m_subject->setGenes(EvolvingStringDNA::RandomString(m_subject->getTarget().size()));
}

EvolvingStringDNA::Fitness EvolvingStringDNA::fitness() const
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

    fitness /= genes.size();

    return fitness;
}

EvolvingString * EvolvingStringDNA::crossover(EvolvingStringDNA const & partner) const
{
    assert(m_subject);

    EvolvingString * child = new EvolvingString(*m_subject);

    std::size_t length = m_subject->getGenes().size();

    std::random_device rd;
    std::mt19937 rng(rd());
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
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<MutationRate> random(0.0, 1.0);

    auto & genes = m_subject->getGenes();
    std::size_t length = genes.size();

    for(auto i = 0u; i < length; ++i)
    {
        if(random(rng) < mutationRate)
        {
            //TODO: optimize this
            genes[i] = RandomString(1)[0];
        }
    }
}


std::string EvolvingStringDNA::RandomString(std::size_t length)
{
    static const char alphabet[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789 ";

    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_int_distribution<> dist(0,sizeof(alphabet)/sizeof(*alphabet)-2);

    std::string str;
    str.reserve(length);
    std::generate_n(std::back_inserter(str), length, [&]() { return alphabet[dist(rng)];});

    return str;
}

void evolutionTest()
{
    Individuals<EvolvingString> strings;
    for(auto i = 0; i < 200; ++i)
    {
        strings.push_back(new EvolvingString("To be or not to be"));
    }

    Population<EvolvingStringDNA> dnas = simulate<EvolvingStringDNA>(strings, 300);

    for(auto & dna: dnas)
    {
        auto subject = dna.getSubject();
        std::cout << subject->getGenes() << std::endl;
    }
}

}
