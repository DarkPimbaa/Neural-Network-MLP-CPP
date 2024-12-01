#include "redeNeural.hpp"
#include <random>
#include <ctime>

RedeNeural::RedeNeural(int entrada, int numLayers, int saida) {
    this->entrada = entrada;
    this->numLayers = numLayers;
    this->saida = saida;
    gerarPesos(entrada, numLayers, saida);
}

RedeNeural::~RedeNeural() {
}

std::vector<double> RedeNeural::gerarPesosAleatorios(int quantidade) {
    std::vector<double> pesos;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    for(int i = 0; i < quantidade; i++) {
        pesos.push_back(dis(gen));
    }
    return pesos;
}

void RedeNeural::gerarPesos(int entrada, int numLayers, int saida) {
    // Criar layer de entrada
    Layer inputLayer;
    for(int i = 0; i < entrada; i++) {
        Neuronio n;
        n.valor = 0;
        inputLayer.neuronios.push_back(n);
    }
    rede.layers.push_back(inputLayer);

    // Criar layers ocultas
    for(int l = 0; l < numLayers; l++) {
        Layer hiddenLayer;
        for(int i = 0; i < entrada; i++) {  // Usando mesmo número de neurônios que entrada
            Neuronio n;
            n.valor = 0;
            // Cada neurônio deve ter um peso para cada neurônio do layer anterior
            n.pesos = gerarPesosAleatorios(entrada);  // Sempre usa o número de entradas
            hiddenLayer.neuronios.push_back(n);
        }
        rede.layers.push_back(hiddenLayer);
    }

    // Criar layer de saída
    Layer outputLayer;
    for(int i = 0; i < saida; i++) {
        Neuronio n;
        n.valor = 0;
        n.pesos = gerarPesosAleatorios(entrada);  // Sempre usa o número de entradas
        outputLayer.neuronios.push_back(n);
    }
    rede.layers.push_back(outputLayer);
}

void RedeNeural::modificarPesos(double valor) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> signal(0, 1);

    for(auto& layer : rede.layers) {
        for(auto& neuronio : layer.neuronios) {
            for(auto& peso : neuronio.pesos) {
                if(dis(gen) < 0.5) {  // 50% de chance de modificar
                    double modificacao = signal(gen) ? valor : -valor;
                    peso += modificacao;
                    
                    // Limita o peso ao intervalo [-1, 1]
                    if(peso > 1.0) peso = 1.0;
                    else if(peso < -1.0) peso = -1.0;
                }
            }
        }
    }
}

Rede RedeNeural::truncamento(std::vector<Rede> redes) {
    if(redes.empty()) return Rede();
    
    Rede mediaRede = redes[0];  // Inicializa com a primeira rede
    
    // Para cada layer
    for(size_t l = 0; l < mediaRede.layers.size(); l++) {
        // Para cada neurônio
        for(size_t n = 0; n < mediaRede.layers[l].neuronios.size(); n++) {
            // Para cada peso
            for(size_t p = 0; p < mediaRede.layers[l].neuronios[n].pesos.size(); p++) {
                double somaPesos = 0;
                // Soma os pesos de todas as redes
                for(const auto& rede : redes) {
                    somaPesos += rede.layers[l].neuronios[n].pesos[p];
                }
                // Calcula a média
                mediaRede.layers[l].neuronios[n].pesos[p] = somaPesos / redes.size();
            }
        }
    }
    
    return mediaRede;
}

std::vector<bool> RedeNeural::iniciar(std::vector<double> inputs) {
    // Verificar se o número de inputs corresponde ao número de neurônios de entrada
    if(inputs.size() != rede.layers[0].neuronios.size()) {
        return std::vector<bool>();  // Retorna vetor vazio em caso de erro
    }

    // Definir valores do layer de entrada
    for(size_t i = 0; i < inputs.size(); i++) {
        rede.layers[0].neuronios[i].valor = inputs[i];
    }

    // Processar layers ocultos e de saída
    for(size_t l = 1; l < rede.layers.size(); l++) {
        for(size_t n = 0; n < rede.layers[l].neuronios.size(); n++) {
            double soma = 0.0;
            
            // Soma ponderada com layer anterior
            for(size_t prev = 0; prev < rede.layers[l-1].neuronios.size(); prev++) {
                soma += rede.layers[l-1].neuronios[prev].valor * 
                       rede.layers[l].neuronios[n].pesos[prev];
            }
            
            // Aplicar ReLU para layers ocultos
            if(l < rede.layers.size() - 1) {
                rede.layers[l].neuronios[n].valor = soma > 0 ? soma : 0;
            } else {
                rede.layers[l].neuronios[n].valor = soma;  // Layer de saída sem ReLU
            }
        }
    }

    // Converter último layer para booleanos e zerar todos os valores
    std::vector<bool> saida;
    for(const auto& neuronio : rede.layers.back().neuronios) {
        saida.push_back(neuronio.valor > 0);
    }

    // Zerar todos os valores dos neurônios
    for(auto& layer : rede.layers) {
        for(auto& neuronio : layer.neuronios) {
            neuronio.valor = 0;
        }
    }
    
    return saida;
}

bool RedeNeural::setRede(Rede rede) {
    this->rede = rede;
    return true;
}