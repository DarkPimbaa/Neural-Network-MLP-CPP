
#include <iostream>
#include <cassert>
#include <iomanip>
#include "redeNeural.hpp"

// Função auxiliar para verificar se um número está no intervalo [-1, 1]
bool estaDentroDoIntervalo(double valor) {
    return valor >= -1.0 && valor <= 1.0;
}

// Função auxiliar para imprimir o resultado de um teste
void imprimirResultadoTeste(const std::string& nomeTeste, bool sucesso) {
    std::cout << "[" << (sucesso ? "PASSOU" : "FALHOU") << "] " << nomeTeste << std::endl;
}

// Testa a criação da rede neural
bool testarCriacaoRede() {
    RedeNeural rede(3, 2, 2);  // 3 entradas, 2 layers ocultos, 2 saídas
    Rede estrutura = rede.getRede();
    
    bool sucesso = true;
    
    // Verifica número de layers (entrada + ocultos + saída)
    if(estrutura.layers.size() != 4) {  // 1 entrada + 2 ocultos + 1 saída
        std::cout << "Número incorreto de layers: " << estrutura.layers.size() << std::endl;
        sucesso = false;
    }
    
    // Verifica número de neurônios em cada layer (incluindo bias)
    if(estrutura.layers[0].neuronios.size() != 4) {  // Layer de entrada (3 + 1 bias)
        std::cout << "Número incorreto de neurônios no layer de entrada: " << estrutura.layers[0].neuronios.size() << " (esperado 4)" << std::endl;
        sucesso = false;
    }
    
    if(estrutura.layers[1].neuronios.size() != 4 ||  // Primeiro layer oculto (3 + 1 bias)
       estrutura.layers[2].neuronios.size() != 4) {  // Segundo layer oculto (3 + 1 bias)
        std::cout << "Número incorreto de neurônios nos layers ocultos" << std::endl;
        sucesso = false;
    }
    
    if(estrutura.layers[3].neuronios.size() != 2) {  // Layer de saída (sem bias)
        std::cout << "Número incorreto de neurônios no layer de saída" << std::endl;
        sucesso = false;
    }

    // Verifica se os neurônios bias têm valor 1.0
    for(size_t l = 0; l < estrutura.layers.size() - 1; l++) {  // -1 para não verificar o layer de saída
        double valorBias = estrutura.layers[l].neuronios.back().valor;
        if(valorBias != 1.0) {
            std::cout << "Valor incorreto do neurônio bias no layer " << l << ": " << valorBias << " (esperado 1.0)" << std::endl;
            sucesso = false;
        }
    }
    
    return sucesso;
}

// Testa se os pesos estão no intervalo correto [-1, 1]
bool testarPesos() {
    RedeNeural rede(3, 2, 2);
    Rede estrutura = rede.getRede();
    
    bool sucesso = true;
    
    // Verifica os pesos de cada layer (exceto o de entrada)
    for(size_t l = 1; l < estrutura.layers.size(); l++) {
        for(const auto& neuronio : estrutura.layers[l].neuronios) {
            for(double peso : neuronio.pesos) {
                if(!estaDentroDoIntervalo(peso)) {
                    std::cout << "Peso fora do intervalo [-1, 1]: " << peso << std::endl;
                    sucesso = false;
                }
            }
        }
    }
    
    return sucesso;
}

// Testa a modificação de pesos
bool testarModificacaoPesos() {
    RedeNeural rede(3, 2, 2);
    
    // Guarda os pesos originais
    Rede estruturaOriginal = rede.getRede();
    
    // Modifica os pesos
    rede.modificarPesos(0.5);
    
    Rede estruturaModificada = rede.getRede();
    bool sucesso = true;
    
    // Verifica se algum peso foi modificado e se todos estão no intervalo
    bool algunsPesosDiferentes = false;
    
    for(size_t l = 1; l < estruturaModificada.layers.size(); l++) {
        for(size_t n = 0; n < estruturaModificada.layers[l].neuronios.size(); n++) {
            // Verifica se o neurônio bias manteve seu valor como 1.0
            if(l < estruturaModificada.layers.size() - 1 && n == estruturaModificada.layers[l].neuronios.size() - 1) {
                if(estruturaModificada.layers[l].neuronios[n].valor != 1.0) {
                    std::cout << "Valor do neurônio bias foi alterado incorretamente" << std::endl;
                    sucesso = false;
                }
                continue;
            }

            for(size_t p = 0; p < estruturaModificada.layers[l].neuronios[n].pesos.size(); p++) {
                double pesoOriginal = estruturaOriginal.layers[l].neuronios[n].pesos[p];
                double pesoModificado = estruturaModificada.layers[l].neuronios[n].pesos[p];
                
                if(pesoOriginal != pesoModificado) {
                    algunsPesosDiferentes = true;
                }
                
                if(!estaDentroDoIntervalo(pesoModificado)) {
                    std::cout << "Peso modificado fora do intervalo [-1, 1]: " << pesoModificado << std::endl;
                    sucesso = false;
                }
            }
        }
    }
    
    if(!algunsPesosDiferentes) {
        std::cout << "Nenhum peso foi modificado!" << std::endl;
        sucesso = false;
    }
    
    return sucesso;
}

// Testa a propagação de valores pela rede
bool testarPropagacao() {
    RedeNeural rede(3, 2, 2);
    std::vector<double> inputs = {0.5, -0.5, 1.0};
    
    std::vector<bool> resultado = rede.iniciar(inputs);
    
    bool sucesso = true;
    
    // Verifica se o número de saídas está correto
    if(resultado.size() != 2) {
        std::cout << "Número incorreto de saídas: " << resultado.size() << std::endl;
        sucesso = false;
    }
    
    // Verifica se os valores dos neurônios foram zerados após a propagação (exceto bias)
    Rede estrutura = rede.getRede();
    for(size_t l = 0; l < estrutura.layers.size(); l++) {
        for(size_t n = 0; n < estrutura.layers[l].neuronios.size(); n++) {
            double valor = estrutura.layers[l].neuronios[n].valor;
            
            // Se é um neurônio bias (último neurônio de cada layer, exceto o de saída)
            bool ehBias = (l < estrutura.layers.size() - 1) && (n == estrutura.layers[l].neuronios.size() - 1);
            
            if(ehBias && valor != 1.0) {
                std::cout << "Valor do neurônio bias foi alterado: " << valor << " (deveria ser 1.0)" << std::endl;
                sucesso = false;
            } else if(!ehBias && valor != 0.0) {
                std::cout << "Valor do neurônio não foi zerado após propagação: " << valor << std::endl;
                sucesso = false;
            }
        }
    }
    
    return sucesso;
}

// Testa o truncamento de redes
bool testarTruncamento() {
    RedeNeural rede1(3, 2, 2);
    RedeNeural rede2(3, 2, 2);
    
    std::vector<Rede> redes = {rede1.getRede(), rede2.getRede()};
    
    Rede redeTruncada = rede1.truncamento(redes);
    bool sucesso = true;
    
    // Verifica se a estrutura da rede truncada está correta
    if(redeTruncada.layers.size() != 4) {
        std::cout << "Número incorreto de layers na rede truncada" << std::endl;
        sucesso = false;
    }
    
    // Verifica se os pesos da rede truncada estão no intervalo [-1, 1]
    for(const auto& layer : redeTruncada.layers) {
        for(const auto& neuronio : layer.neuronios) {
            for(double peso : neuronio.pesos) {
                if(!estaDentroDoIntervalo(peso)) {
                    std::cout << "Peso da rede truncada fora do intervalo [-1, 1]: " << peso << std::endl;
                    sucesso = false;
                }
            }
        }
    }
    
    return sucesso;
}

int main() {
    std::cout << "Iniciando testes da Rede Neural..." << std::endl << std::endl;
    
    // Executa todos os testes
    imprimirResultadoTeste("Teste de Criação da Rede", testarCriacaoRede());
    imprimirResultadoTeste("Teste de Pesos", testarPesos());
    imprimirResultadoTeste("Teste de Modificação de Pesos", testarModificacaoPesos());
    imprimirResultadoTeste("Teste de Propagação", testarPropagacao());
    imprimirResultadoTeste("Teste de Truncamento", testarTruncamento());
    
    return 0;
}