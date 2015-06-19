#ifndef NEURO_CAR_EVOLVING_STRING
#define NEURO_CAR_EVOLVING_STRING

#include <string>

#include <dna.hpp>
#include <evolution.hpp>

namespace NeuroCar {

struct EvolvingString
{
    EvolvingString(std::string const & target):
        m_genes(""),
        m_target(target)
    {

    }

    std::string & getGenes()
    {
        return m_genes;
    }

    std::string const & getGenes() const
    {
        return m_genes;
    }

    std::string const & getTarget() const
    {
        return m_target;
    }

    void setGenes(std::string const & genes)
    {
        m_genes = genes;
    }

    private:
        std::string m_genes;
        std::string m_target;
};

class EvolvingStringDNA : public DNA<EvolvingString, EvolvingStringDNA>
{
    public:
        EvolvingStringDNA(EvolvingString * subject = nullptr);

        virtual void randomize() override;
        virtual Fitness computeFitness() override;
        virtual Fitness getFitness() const override;
        virtual EvolvingString * crossover(EvolvingStringDNA const & partner) const override;
        virtual void mutate(MutationRate mutationRate) override;

        static char RandomChar();
        static std::string RandomString(std::size_t length);

    private:
        Fitness m_fitness;

};

class Random
{
    public:
};

void stringEvolution();

}

#endif //NEURO_CAR_EVOLVING_STRING
