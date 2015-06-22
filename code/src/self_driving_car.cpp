#include <self_driving_car.hpp>

#include <random>

namespace NeuroCar {

SelfDrivingCar::SelfDrivingCar():
    m_car(nullptr),
    m_neuroController()
{
    //m_car->setController(&m_neuroController);
}

SelfDrivingCar::SelfDrivingCar(NeuralNetwork const & nn):
    m_car(nullptr),
    m_neuroController(nn)
{
    //m_car->setController(&m_neuroController);
}

NeuroController & SelfDrivingCar::getNeuroController()
{
    return m_neuroController;
}

NeuroController const & SelfDrivingCar::getNeuroController() const
{
    return m_neuroController;
}

SelfDrivingCar::NeuralNetwork & SelfDrivingCar::getNeuralNetwork()
{
    return m_neuroController.getNeuralNetwork();
}

SelfDrivingCar::NeuralNetwork const & SelfDrivingCar::getNeuralNetwork() const
{
    return m_neuroController.getNeuralNetwork();
}

SelfDrivingCarDNA::SelfDrivingCarDNA(Subject subject):
    DNA(subject),
    m_fitness(0.0)
{

}

void SelfDrivingCarDNA::randomize()
{
    SelfDrivingCar * car = this->getSubject();
    NeuroController & nc = car->getNeuroController();
    SelfDrivingCar::NeuralNetwork nn = nc.getNeuralNetwork();
    nn.synthetize();
}

SelfDrivingCarDNA::Fitness SelfDrivingCarDNA::computeFitness()
{
    Fitness fitness = 0.0;

    #if 0
    SelfDrivingCar * car = this->getSubject();
    vec3 pos = car->drive();
    fitness = distance(pos, destination);
    #endif

    m_fitness = fitness;

    return fitness;
}

SelfDrivingCarDNA::Fitness SelfDrivingCarDNA::getFitness() const
{
    return m_fitness;
}

SelfDrivingCarDNA::Subject SelfDrivingCarDNA::crossover(SelfDrivingCarDNA const & partner) const
{
    using NeuralNetwork = SelfDrivingCar::NeuralNetwork;

    assert(m_subject);

    Subject child(m_subject);

    static std::random_device rd;
    static std::default_random_engine rng(rd());
    std::uniform_real_distribution<double> random(0, 1.0);

    NeuralNetwork const & parentAGenes = m_subject->getNeuralNetwork();
    NeuralNetwork const & parentBGenes = partner.getSubject()->getNeuralNetwork();

    auto const selectParent = [&parentAGenes, &parentBGenes](float r)
        -> NeuralNetwork const &
    {
        return r < 0.5 ? parentAGenes : parentBGenes;
    };

    auto & childGenes = child->getNeuralNetwork();

    // Generate new ADN by combining the parents' DNAs
    auto const & nnshape = parentAGenes.getShape();
    for(auto l = 1u; l < nnshape.size(); ++l)
    {
        auto J = nnshape[l];
        auto I = nnshape[l-1];

        for(auto i = 0u; i < I; ++i)
        {
            for(auto j = 0u; j < J; ++j)
            {
                NeuralNetwork const & nn = selectParent(random(rng));
                childGenes.setWeight(l, i, j, nn.getWeight(l, i, j));
            }

            NeuralNetwork const & nn = selectParent(random(rng));
            childGenes.setBias(l, i, nn.getBias(l, i));
        }
    }

    return child;
}

void SelfDrivingCarDNA::mutate(MutationRate mutationRate)
{
    static std::random_device rd;
    static std::default_random_engine rng(rd());
    std::uniform_real_distribution<MutationRate> random(0.00001, 1.0);

    SelfDrivingCar * car = this->getSubject();
    assert(car);

    auto & nn = car->getNeuralNetwork();

    auto const & nnshape = nn.getShape();
    for(auto l = 1u; l < nnshape.size(); ++l)
    {
        auto J = nnshape[l];
        auto I = nnshape[l-1];

        for(auto i = 0u; i < I; ++i)
        {
            for(auto j = 0u; j < J; ++j)
            {
                MutationRate r = random(rng);
                if(r < mutationRate)
                {
                    auto w = nn.getWeight(l, i, j) * (r / mutationRate);
                    nn.setWeight(l, i, j, w);
                }
            }

            MutationRate r = random(rng);
            if(r < mutationRate)
            {
                auto b = nn.getBias(l, i) * (r / mutationRate);
                nn.setBias(l, i, b);
            }
        }
    }
}

}
