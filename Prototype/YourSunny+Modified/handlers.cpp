#include "WifiCam.hpp"
#include <StreamString.h>
#include <uri/UriBraces.h>
#include <WiFiClient.h> 

static const char FRONTPAGE[] = R"EOT(
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>esp32cam WifiCam Example</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 1rem;
      background-color: #f4f4f9;
      color: #333;
    }
    h1 {
      text-align: center;
      color: #0066cc;
    }
    table {
      width: 100%;
      max-width: 600px;
      margin: 1rem auto;
      border: solid 1px #000000;
      border-collapse: collapse;
    }
    th, td {
      padding: 0.4rem;
      text-align: center;
      border: solid 1px #000000;
    }
    a {
      text-decoration: none;
      color: #0066cc;
    }
    footer {
      margin-top: 1rem;
      text-align: center;
      font-size: 0.9rem;
      color: #777;
    }
    footer a {
      color: #0066cc;
    }
  </style>
</head>
<body>
  <h1>esp32cam WifiCam Example</h1>
  <table>
    <thead>
      <tr><th>BMP</th><th>JPG</th><th>MJPEG</th></tr>
    </thead>
    <tbody id="resolutions">
      <tr><td colspan="3">loading</td></tr>
    </tbody>
  </table>
  <footer>
    Powered by <a href="https://esp32cam.yoursunny.dev/">esp32cam</a><br>
    If you want to watch anime, visit <a href="https://rioanime-play.blogspot.com">RioAnime</a>
  </footer>
  <script type="module">
    async function fetchText(uri, init) {
      const response = await fetch(uri, init);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return (await response.text()).trim().replaceAll("\r\n", "\n");
    }

    try {
      const list = (await fetchText("/resolutions.csv")).split("\n");
      document.querySelector("#resolutions").innerHTML = list.map((r) => `<tr>${
        ["bmp", "jpg", "mjpeg"].map((fmt) => `<td><a href="/${r}.${fmt}">${r}</a></td>`).join("")
      }</tr>`).join("");
    } catch (err) {
      document.querySelector("#resolutions td").textContent = err.toString();
    }
  </script>
</body>
</html>
)EOT";

static void
serveStill(bool wantBmp) {
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("capture() failure");
    server.send(500, "text/plain", "still capture error\n");
    return;
  }
  Serial.printf("capture() success: %dx%d %zub\n", frame->getWidth(), frame->getHeight(),
                frame->size());

  if (wantBmp) {
    if (!frame->toBmp()) {
      Serial.println("toBmp() failure");
      server.send(500, "text/plain", "convert to BMP error\n");
      return;
    }
    Serial.printf("toBmp() success: %dx%d %zub\n", frame->getWidth(), frame->getHeight(),
                  frame->size());
  }

  server.setContentLength(frame->size());
  server.send(200, wantBmp ? "image/bmp" : "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

static void
serveMjpeg() {
  Serial.println("MJPEG streaming begin");
  WiFiClient client = server.client();
  auto startTime = millis();
  int nFrames = esp32cam::Camera.streamMjpeg(client);
  auto duration = millis() - startTime;
  Serial.printf("MJPEG streaming end: %dfrm %0.2ffps\n", nFrames, 1000.0 * nFrames / duration);
}

void
addRequestHandlers() {
  server.on("/", HTTP_GET, [] {
    server.setContentLength(sizeof(FRONTPAGE));
    server.send(200, "text/html");
    server.sendContent(FRONTPAGE, sizeof(FRONTPAGE));
  });

  server.on("/robots.txt", HTTP_GET,
            [] { server.send(200, "text/html", "User-Agent: *\nDisallow: /\n"); });

  server.on("/resolutions.csv", HTTP_GET, [] {
    StreamString b;
    for (const auto& r : esp32cam::Camera.listResolutions()) {
      b.println(r);
    }
    server.send(200, "text/csv", b);
  });

  server.on(UriBraces("/{}x{}.{}"), HTTP_GET, [] {
    long width = server.pathArg(0).toInt();
    long height = server.pathArg(1).toInt();
    String format = server.pathArg(2);
    if (width == 0 || height == 0 || !(format == "bmp" || format == "jpg" || format == "mjpeg")) {
      server.send(404);
      return;
    }

    auto r = esp32cam::Camera.listResolutions().find(width, height);
    if (!r.isValid()) {
      server.send(404, "text/plain", "non-existent resolution\n");
      return;
    }
    if (r.getWidth() != width || r.getHeight() != height) {
      server.sendHeader("Location",
                        String("/") + r.getWidth() + "x" + r.getHeight() + "." + format);
      server.send(302);
      return;
    }

    if (!esp32cam::Camera.changeResolution(r)) {
      Serial.printf("changeResolution(%ld,%ld) failure\n", width, height);
      server.send(500, "text/plain", "changeResolution error\n");
    }
    Serial.printf("changeResolution(%ld,%ld) success\n", width, height);

    if (format == "bmp") {
      serveStill(true);
    } else if (format == "jpg") {
      serveStill(false);
    } else if (format == "mjpeg") {
      serveMjpeg();
    }
  });
}
