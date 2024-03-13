// GLFWWindowManager.h
#ifndef GLFW_WINDOW_MANAGER_H
#define GLFW_WINDOW_MANAGER_H

#include <GLFW/glfw3.h>
#include <memory>

class GLFWWindowManager {
public:
    static GLFWWindowManager& getInstance() {
        static GLFWWindowManager instance;
        return instance;
    }

    GLFWwindow* getWindow() { return window; }

    // Prevent copying and assignment
    GLFWWindowManager(const GLFWWindowManager&) = delete;
    GLFWWindowManager& operator=(const GLFWWindowManager&) = delete;

private:
    GLFWwindow* window;
    GLFWWindowManager();
    ~GLFWWindowManager();

    void initializeGLFW();
    void createWindow();
    void destroyWindow();
};

#endif // GLFW_WINDOW_MANAGER_H
