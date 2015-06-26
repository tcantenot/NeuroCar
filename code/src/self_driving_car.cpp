#include <self_driving_car.hpp>
#include <renderer.hpp>

#include <random>

namespace NeuroCar {

SelfDrivingCar::SelfDrivingCar():
    m_car(nullptr),
    m_neuroController()
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

void SelfDrivingCar::setNeuroController(NeuroController nc)
{
    m_neuroController = nc;
}

SelfDrivingCar::NeuralNetwork const & SelfDrivingCar::getNeuralNetwork() const
{
    return m_neuroController.getNeuralNetwork();
}


b2Vec2 const & SelfDrivingCar::getDestination() const
{
    return m_destination;
}

void SelfDrivingCar::setDestination(b2Vec2 destination)
{
    m_destination = destination;
}

void SelfDrivingCar::setWorldSeed(uint32_t seed)
{
    m_worldSeed = seed;
}

uint32_t SelfDrivingCar::getWorldSeed()
{
    return m_worldSeed;
}


std::shared_ptr<Car> & SelfDrivingCar::getCar()
{
    return m_car;
}

std::shared_ptr<Car> const & SelfDrivingCar::getCar() const
{
    return m_car;
}

void SelfDrivingCar::setCar(std::shared_ptr<Car> car)
{
    m_car = car;
    m_car->setController(&m_neuroController);
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

    std::shared_ptr<Car> car = this->getSubject()->getCar();

    // Create world
    uint32_t worldWidth = 100;
    uint32_t worldHeight = 80;

    #if CAR_PHYSICS_GRAPHIC_MODE_SFML
    Renderer r(8, worldWidth, worldHeight);
    World *w = new World(8, 3, &r);
    #else
    World *w = new World(8, 3);
    #endif


    w->addBorders(worldWidth, worldHeight);

    uint32_t seed = this->getSubject()->getWorldSeed();

    w->randomize(worldWidth, worldHeight, 15, seed);

    while (w->willCollide(car))
    {
        delete w;
        #if CAR_PHYSICS_GRAPHIC_MODE_SFML
        w = new World(8, 3, &r);
        #else
        w = new World(8, 3);
        #endif
        w->randomize(worldWidth, worldHeight, 15, ++seed);
    }

    w->addRequiredDrawable(car);

    w->run();

    delete w;

    b2Vec2 pos = car->getPos();

    b2Vec2 destination = this->getSubject()->getDestination();

    fitness = 1.0/(pow((pos.x - destination.x), 2) + pow((pos.y - destination.y), 2));

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
