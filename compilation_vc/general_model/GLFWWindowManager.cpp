// GLFWWindowManager.cpp
#include "GLFWWindowManager.h"
#include <stdexcept>

GLFWWindowManager::GLFWWindowManager() {
    initializeGLFW();
    createWindow();
}

GLFWWindowManager::~GLFWWindowManager() {
    destroyWindow();
}

void GLFWWindowManager::initializeGLFW() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }
}

void GLFWWindowManager::createWindow() {
    window = glfwCreateWindow(1200, 900, "MuJoCo Environment", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window.");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
}

void GLFWWindowManager::destroyWindow() {
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}
