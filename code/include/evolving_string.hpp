#ifndef EVOLVING_STRING
#define EVOLVING_STRING

#include <string>

#include <dna.hpp>
#include <evolution.hpp>

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
        EvolvingStringDNA();

        virtual void init(Params const & params) override;
        virtual void randomize(std::size_t seed) override;
        virtual Fitness computeFitness(std::size_t ngen = 0) override;
        virtual void reset() override;
        virtual Subject crossover(EvolvingStringDNA const & partner) const override;
        virtual void mutate(MutationRate mutationRate) override;

        static char RandomChar();
        static std::string RandomString(std::size_t length);
};

void stringEvolution();

#endif //EVOLVING_STRING
