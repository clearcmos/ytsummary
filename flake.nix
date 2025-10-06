{
  description = "AI-powered YouTube video summarizer with RAG-enhanced Q&A";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          requests
          numpy
          scikit-learn
          sentence-transformers
          rich
          fastapi
          uvicorn
          pydantic
        ]);

        ytsummary = pkgs.stdenv.mkDerivation {
          pname = "ytsummary";
          version = "1.0.0";

          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];

          installPhase = ''
            mkdir -p $out/bin
            mkdir -p $out/share/ytsummary
            mkdir -p $out/share/ytsummary/static

            # Install Python scripts
            cp download_subs.py $out/share/ytsummary/
            cp app.py $out/share/ytsummary/

            # Install static web files
            cp -r static/* $out/share/ytsummary/static/

            # Create CLI wrapper
            makeWrapper ${pythonEnv}/bin/python $out/bin/ytsummary \
              --add-flags "$out/share/ytsummary/download_subs.py" \
              --prefix PATH : ${pkgs.yt-dlp}/bin

            # Create web server wrapper
            makeWrapper ${pythonEnv}/bin/python $out/bin/ytsummary-web \
              --add-flags "-m uvicorn app:app --host 0.0.0.0" \
              --set PYTHONPATH "$out/share/ytsummary" \
              --prefix PATH : ${pkgs.yt-dlp}/bin \
              --run "cd $out/share/ytsummary"
          '';

          meta = with pkgs.lib; {
            description = "AI-powered YouTube video summarizer with RAG";
            license = licenses.mit;
            maintainers = [ ];
          };
        };

      in {
        packages.default = ytsummary;
        packages.ytsummary = ytsummary;

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.yt-dlp
          ];
        };
      }
    ) // {
      nixosModules.default = { config, lib, pkgs, ... }:
        with lib;
        let
          cfg = config.services.ytsummary;
        in {
          options.services.ytsummary = {
            enable = mkEnableOption "YouTube Summary AI service";

            port = mkOption {
              type = types.port;
              default = 8000;
              description = "Port for the web interface";
            };

            ollamaUrl = mkOption {
              type = types.str;
              default = "http://localhost:11434";
              description = "Ollama API URL";
            };

            dataDir = mkOption {
              type = types.path;
              default = "/var/lib/ytsummary";
              description = "Directory for storing downloaded subtitles";
            };

            retention = {
              enabled = mkOption {
                type = types.bool;
                default = true;
                description = "Enable automatic cleanup of old subtitle files";
              };

              maxAge = mkOption {
                type = types.str;
                default = "30d";
                description = "Maximum age of subtitle files before cleanup (systemd time format)";
              };
            };

            user = mkOption {
              type = types.str;
              default = "ytsummary";
              description = "User account under which ytsummary runs";
            };

            group = mkOption {
              type = types.str;
              default = "ytsummary";
              description = "Group under which ytsummary runs";
            };
          };

          config = mkIf cfg.enable {
            users.users.${cfg.user} = {
              isSystemUser = true;
              group = cfg.group;
              description = "YouTube Summary service user";
              home = cfg.dataDir;
              createHome = true;
            };

            users.groups.${cfg.group} = {};

            systemd.services.ytsummary = {
              description = "YouTube Summary AI Web Service";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];
              wants = [ "ollama.service" ];

              environment = {
                OLLAMA_HOST = cfg.ollamaUrl;
                YTSUMMARY_DATA_DIR = cfg.dataDir;
              };

              serviceConfig = {
                Type = "simple";
                User = cfg.user;
                Group = cfg.group;
                WorkingDirectory = cfg.dataDir;
                ExecStart = "${self.packages.${pkgs.system}.ytsummary}/bin/ytsummary-web --port ${toString cfg.port}";
                Restart = "on-failure";
                RestartSec = "10s";

                # Security hardening
                NoNewPrivileges = true;
                PrivateTmp = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                ReadWritePaths = [ cfg.dataDir ];
              };
            };

            # Cleanup timer for old subtitle files
            systemd.services.ytsummary-cleanup = mkIf cfg.retention.enabled {
              description = "Cleanup old YouTube subtitle files";
              serviceConfig = {
                Type = "oneshot";
                User = cfg.user;
                Group = cfg.group;
                ExecStart = "${pkgs.findutils}/bin/find ${cfg.dataDir} -name '*.srt' -type f -mtime +${cfg.retention.maxAge} -delete";
              };
            };

            systemd.timers.ytsummary-cleanup = mkIf cfg.retention.enabled {
              description = "Timer for YouTube subtitle cleanup";
              wantedBy = [ "timers.target" ];
              timerConfig = {
                OnCalendar = "daily";
                Persistent = true;
              };
            };
          };
        };
    };
}
