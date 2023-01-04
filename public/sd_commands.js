for (const funcName in window.api) {
    window[funcName] = async (...args) => window.api[funcName](funcName, ...args);
}
