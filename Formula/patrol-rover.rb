class PatrolRover < Formula
  desc "Autonomous AI Vision Patrol Rover with 3D Chase Camera & GTA Minimap HUD"
  homepage "https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover"
  url "https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover/archive/refs/heads/main.tar.gz"
  version "1.0.0"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Autonomous Patrol Rover", shell_output("#{bin}/patrol-rover --help")
  end
end
