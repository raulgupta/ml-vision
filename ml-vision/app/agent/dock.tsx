import React from "react";
import { FloatingDock } from "../../components/FloatingDock";
import Image from "next/image";

export function FloatingDockDemo() {
  const links = [
    {
      title: "T. Shelby",
      icon: (
        <Image
          src="/shelby2.png"
          width={30}
          height={30}
          className="h-full w-full object-cover rounded-full scale-150"
          alt="Avengers"
        />  
      ),
      href: "#",
    },
    {
      title: "R. Fineman",
      icon: (
        <Image
          src="/fineman.png"
          width={30}
          height={30}
          className="h-full w-full object-cover rounded-full scale-150"
          alt="Avengers"
        />
      ),
      href: "#",
    },
    {
      title: "N. Ravikant",
      icon: (
        <Image
          src="/naval.png"
          width={30}
          height={30}
          className="h-full w-full object-cover rounded-full scale-150"
          alt="Avengers"
        />
      ),
      href: "#",
    },
    {
      title: "Fleux",
      icon: (
        <Image
          src="/fleux-lg.png"
          width={190}
          height={190}
          alt="Avengers"
          className="h-full w-full object-cover"
        />
      ),
      href: "#",
    },
    {
      title: "J. Bezos",
      icon: (
        <Image
          src="/bezos.png"
          width={30}
          height={30}
          className="h-full w-full object-cover rounded-full scale-150"
          alt="Avengers"
        />  
      ),
      href: "#",
    },
    {
      title: "S. Jobs",
      icon: (
        <Image
          src="/jobs.png"
          width={30}
          height={30}
          className="h-full w-full object-cover rounded-full scale-150"
          alt="Avengers"
        />
      ),
      href: "#",
    },
    {
      title: "E. Musk",
      icon: (
        <Image
          src="/musk2.png"
          width={30}
          height={30}
          className="h-full w-full object-cover rounded-full scale-150"
          alt="Avengers"
        />
      ),
      href: "#",
    },
  ];
  return (
    <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg shadow-[0_0_15px_rgba(255,255,255,0.02)] p-8">
      <div className="flex items-center justify-center h-[35rem] md:h-[20rem] w-full">
        <FloatingDock
          mobileClassName="translate-y-20"
          items={links}
          desktopClassName="scale-150"
        />
      </div>
    </div>
  );
}
