import React from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/lib/utils";
import UrdfViewer from "../UrdfViewer";
import Logo from "@/components/Logo";
import type { DefaultRobotType } from "@/lib/defaultUrdfModels";

export interface BimanualViewerSpec {
  robotType: DefaultRobotType;
  jointPrefix: string;
  label: string;
}

interface VisualizerPanelProps {
  onGoBack: () => void;
  className?: string;
  /** Optional content rendered as a column beside the 3D viewer (e.g. a camera panel). */
  rightSlot?: React.ReactNode;
  /** Bimanual only: renders a labeled left/right pair of viewers instead of
   * the single context-driven one. */
  bimanualViewers?: [BimanualViewerSpec, BimanualViewerSpec];
}

const VisualizerPanel: React.FC<VisualizerPanelProps> = ({
  onGoBack,
  className,
  rightSlot,
  bimanualViewers,
}) => {
  return (
    <div
      className={cn(
        "w-full p-2 sm:p-4 space-y-4 lg:space-y-0 lg:space-x-4 flex flex-col lg:flex-row",
        className
      )}
    >
      <div className="bg-gray-900 rounded-lg p-4 flex-1 flex flex-col">
        <div className="flex items-center gap-4 mb-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={onGoBack}
            className="text-gray-400 hover:text-white hover:bg-gray-800 flex-shrink-0"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <Logo iconOnly={true} />
          <div className="w-px h-6 bg-gray-700" />
          <h2 className="text-xl font-medium text-gray-200">Teleoperation</h2>
        </div>
        <div className="flex-1 bg-black rounded border border-gray-800 min-h-[50vh] lg:min-h-0">
          {bimanualViewers ? (
            <div className="w-full h-full flex flex-col md:flex-row gap-2">
              {bimanualViewers.map((v) => (
                <div key={v.label} className="flex-1 flex flex-col min-h-[25vh] md:min-h-0">
                  <div className="text-xs text-gray-400 px-2 pt-1">{v.label}</div>
                  <div className="flex-1">
                    <UrdfViewer robotType={v.robotType} jointPrefix={v.jointPrefix} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <UrdfViewer />
          )}
        </div>
      </div>
      {rightSlot && (
        <div className="lg:w-96 flex flex-col">{rightSlot}</div>
      )}
    </div>
  );
};

export default VisualizerPanel;
