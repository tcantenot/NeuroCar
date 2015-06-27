#include <self_driving_car.hpp>
#include <renderer.hpp>

#include <random>

#include <iostream>

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
    m_neuroController.setDestination(destination);
}

void SelfDrivingCar::setWorldSeed(uint32_t seed)
{
    m_worldSeed = seed;
}

uint32_t SelfDrivingCar::getWorldSeed()
{
    return m_worldSeed;
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

void SelfDrivingCarDNA::randomize(std::size_t seed)
{
    auto car = this->getSubject();
    NeuroController & nc = car->getNeuroController();
    SelfDrivingCar::NeuralNetwork & nn = nc.getNeuralNetwork();
    nn.setSeed(seed);
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

    //w->randomize(worldWidth, worldHeight, 15, seed);

    /*while (w->willCollide(car))
    {
        delete w;
        #if CAR_PHYSICS_GRAPHIC_MODE_SFML
        w = new World(8, 3, &r);
        #else
        w = new World(8, 3);
        #endif
        w->addBorders(worldWidth, worldHeight);
        //w->randomize(worldWidth, worldHeight, 15, ++seed);
    }*/

    w->addRequiredDrawable(car);
    if(0)
    {
        std::cout << "World " << w << std::endl;
        auto car = m_subject->getCar();
        auto body = car->getBody();
        auto v = body->GetLinearVelocity();
        std::cout << "Body linear velocity: " << v.x << ", " << v.y << std::endl;
        auto p = body->GetPosition();
        std::cout << "Body position: " << v.x << ", " << v.y << std::endl;
        std::cout << "Body mass: " << body->GetMass() << std::endl;
    }


    if(0)
    {
        b2Vec2 pos = car->getPos();
        //std::cout << "Body: " << car->getBody() << std::endl;
        std::cout << pos.x << " " << pos.y << std::endl;
        //std::cout << "Angle: "  << car->getAngle() << std::endl;
        b2Vec2 destination = this->getSubject()->getDestination();
        //std::cout << "Destination: "  << destination.x << ", " << destination.y << std::endl;
    }

    w->run();

    if(0)
    {
        std::cout << "###" << std::endl;
        //std::cout << "Body: " << car->getBody() << std::endl;
        b2Vec2 pos = car->getPos();
        std::cout << pos.x << " " << pos.y << std::endl;
        //std::cout << "Angle: "  << car->getAngle() << std::endl;
        b2Vec2 destination = this->getSubject()->getDestination();
        //std::cout << "Destination: "  << destination.x << ", " << destination.y << std::endl;
    }

    b2Vec2 pos = car->getPos();
    //std::cout << pos.x << " " << pos.y << std::endl;

    b2Vec2 destination = this->getSubject()->getDestination();

    fitness = 100 - sqrt(std::pow((pos.x - destination.x), 2) + std::pow((pos.y - destination.y), 2));

    m_fitness = fitness;

    delete w;

    return fitness;
}

SelfDrivingCarDNA::Fitness SelfDrivingCarDNA::getFitness() const
{
    return m_fitness;
}

void SelfDrivingCarDNA::reset()
{
    std::cout << "Reset" << std::endl;

    // Copy and reset previous car
    m_subject->setCar(std::make_shared<Car>(*m_subject->getCar()));

    std::cout << *(m_subject->getCar()) << std::endl;

}

SelfDrivingCarDNA::Subject SelfDrivingCarDNA::crossover(SelfDrivingCarDNA const & partner) const
{
    using NeuralNetwork = SelfDrivingCar::NeuralNetwork;

    assert(m_subject);

    Subject child = createIndividual<NeuroCar::SelfDrivingCar>(*m_subject);

    // Copy and reset previous car
    child->setCar(std::make_shared<Car>(*m_subject->getCar()));

    std::cout << "Crossover" << std::endl;
    std::cout << *(child->getCar()) << std::endl;

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
    for(auto l = 0u; l < nnshape.size()-1; ++l)
    {
        auto I = nnshape[l];
        auto J = nnshape[l+1];

        for(auto j = 0u; j < J; ++j)
        {
            for(auto i = 0u; i < I; ++i)
            {
                NeuralNetwork const & nn = selectParent(random(rng));
                childGenes.setWeight(l, i, j, nn.getWeight(l, i, j));
            }

            NeuralNetwork const & nn = selectParent(random(rng));
            // j because the the weight are stored in transpose
            childGenes.setBias(l, j, nn.getBias(l, j));
        }
    }

    return child;
}

void SelfDrivingCarDNA::mutate(MutationRate mutationRate)
{
    static std::random_device rd;
    static std::default_random_engine rng(rd());
    std::uniform_real_distribution<double> lottery(0.0, 1.0);
    std::uniform_real_distribution<MutationRate> mutation(-1.0, 1.0);

    auto car = this->getSubject();
    assert(car);

    auto & nn = car->getNeuralNetwork();

    auto const & nnshape = nn.getShape();
    for(auto l = 0u; l < nnshape.size()-1; ++l)
    {
        auto I = nnshape[l];
        auto J = nnshape[l+1];

        for(auto j = 0u; j < J; ++j)
        {
            for(auto i = 0u; i < I; ++i)
            {
                MutationRate r = lottery(rng);
                if(r < mutationRate)
                {
                    auto w = nn.getWeight(l, i, j) + mutation(rng);
                    nn.setWeight(l, i, j, w);
                }
            }

            MutationRate r = lottery(rng);
            if(r < mutationRate)
            {
                // j because the the weight are stored in transpose
                auto b = nn.getBias(l, j) + mutation(rng);
                nn.setBias(l, j, b);
            }
        }
    }
}


}
