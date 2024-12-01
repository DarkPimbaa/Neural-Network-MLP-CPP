#pragma once
#include <vector>

struct Neuronio {
    double valor;
    std::vector<double> pesos;  
};

struct Layer {
    std::vector<Neuronio> neuronios;
};

struct Rede {
    std::vector<Layer> layers;
};

class RedeNeural {
private:
    Rede rede;
    int entrada;
    int numLayers;
    int saida;
public:
    RedeNeural(int entrada, int numLayers, int saida);
    ~RedeNeural();

    /** Função que inicia os calculos da rede */
    std::vector<bool> iniciar(std::vector<double> inputs);
    
    /** Função que vai gerar pesos aleatórios entre -1 e 1 para todos os neurônios da rede */
    void gerarPesos(int entrada, int numLayers, int saida);

    /** Função auxiliar para gerarPesos() retorna um vector com um numeros de indices igual a entrada, todos com pesos aleatórios entre -1 e 1 */
    std::vector<double> gerarPesosAleatorios(int entrada);

    /** Função que vai modificar os pesos da rede para treinamento, pega o valor que é um número positivo, aleatoriamente torna esse valor positivo ou negativo, e a cada neuronio, tem 50% de chance de ser alterado
    * @param valor - valor que vai ser modificado
    */
    void modificarPesos(double valor);

    /** Função de truncamento, recebe um vector de Rede, pega todos os pesos de todas as redes, e retorna uma rede cujos pesos são a média dos pesos de todas as redes */
    Rede truncamento(std::vector<Rede> redes);

    /** Função que retorna a rede gerada */
    Rede getRede(){
        return this->rede;
    }

    /** Função que seta a rede */
    bool setRede(Rede rede);
};