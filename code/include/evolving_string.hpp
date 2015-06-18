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
        EvolvingStringDNA(EvolvingString * subject);

        virtual void randomize() override;
        virtual Fitness fitness() const override;
        virtual EvolvingString * crossover(EvolvingStringDNA const & partner) const override;
        virtual void mutate(MutationRate mutationRate) override;

        static std::string RandomString(std::size_t length);
};

void evolutionTest();

}

#endif //NEURO_CAR_EVOLVING_STRING
