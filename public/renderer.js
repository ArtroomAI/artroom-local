window.addEventListener("DOMContentLoaded", () => {
  const minimizeButton = document.getElementById("minimize-btn");
  const maxUnmaxButton = document.getElementById("max-unmax-btn");
  const closeButton = document.getElementById("close-btn");

  //TODO Version not working
  //console.log(window.api.getVersion());
  // document.getElementById("artroom-head") = "ArtroomAI v" + window.api.getVersion(); 

  minimizeButton.addEventListener("click", e => {
    window.api.minimizeWindow();
  });

  maxUnmaxButton.addEventListener("click", e => {

    const icon = maxUnmaxButton.querySelector("i.far");

    window.api.maxUnmaxWindow();

    // Change the middle maximize-unmaximize icons.
    if (window.api.isWindowMaximized()) {
      icon.classList.remove("fa-square");
      icon.classList.add("fa-clone");
    } else {
      icon.classList.add("fa-square");
      icon.classList.remove("fa-clone");
    }
  });

  closeButton.addEventListener("click", e => {
    window.api.closeWindow();
  });
});