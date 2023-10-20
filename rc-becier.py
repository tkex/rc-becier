

import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# **********************
# ** KONFIGURATIONEN  **
# **********************

# ** DATEIEN **
# Unterordner in dem die Strecken liegen
UNTERORDNER_STRECKEN = "strecken"

# Streckennamen (an Auskommentieren achten, sofern andere Strecke wie _DizzyButterfly verwendet werden will)
#STRECKE = "_DizzyButterfly.trk"  # oder andere Strecken, die im strecken-Unterordner sind
STRECKE = 'tobiko_2023_torus.csv'

# ** Startpunkte der Visualisierungen **
# Bezierkurve und Diagramme an globalen Punkt t anzeigen
t_wert_global = 0.5
# Frenet-Serret und Parallelrahmen (Bishop) an globalen Punkt t anzeigen
t_wert_frenet_parallel = 0.0

# ** Auflösung der Grafiken **
# Achtung: Die Umstellung auf einen geringen Wert kann dazu führen, dass der berechnete Wert leicht von der visuellen Darstellung abweicht.
# Je höher, desto präziser die Grafik (da mehr Punkte zwischen dem t-Intervall 0 und 1 berechnet werden)
AUFLOESUNG_GRAFIK = 2500  # Standard 2500


class LeseDatei:
    """
    Klasse für das Einlesen der Stützstellen x0 und x3 von .trk und .csv-Dateien.

    Es können jeweils die vorliegenden und vorgefertigen TRK-Dateien eingelesen werden, die anschließend geparsed werden
    um die notwendigen Stützstellen ohne die Sonderzeichen einzulesen. Für die CSV-Dateien werden die x, y, z-Koordinaten
    ebenfalls eingelesen - das Format der CSV-Datei muss dabei x; y; z sein.

    Mittels Fall-Unterscheidung in der trackpunkte_einlesen() wird sichergestellt, welche Parsing-Funktion verwendet wird.
    """

    @staticmethod
    def datei_lesen(dateiname):
        try:
            with open(dateiname, 'r') as datei:
                zeilen = datei.readlines()
            return zeilen
        except FileNotFoundError:
            print(f"Datei {dateiname} nicht gefunden.")
            return []

    @staticmethod
    def trk_daten_parsen(zeilen):
        stuetzpunkte = []
        naechste_zeile_ueberspringen = False
        for zeile in zeilen:
            zeile = zeile.strip()
            if zeile.startswith('#') or zeile == '' or zeile.lower() == 'track':
                naechste_zeile_ueberspringen = True
                continue
            if naechste_zeile_ueberspringen:
                naechste_zeile_ueberspringen = False
                continue

            x, y, z = map(float, zeile.split())
            stuetzpunkte.append((x, y, z))

        return np.array(stuetzpunkte)

    @staticmethod
    def csv_daten_parsen(zeilen):
        stuetzpunkte = []
        for zeile in zeilen:
            zeile = zeile.strip()
            if zeile.startswith('#') or zeile == '':
                continue

            x, y, z = map(float, zeile.split(';'))  # Für andere Trennzeichen, hier anpassen
            stuetzpunkte.append((x, y, z))

        return np.array(stuetzpunkte)

    def trackpunkte_einlesen(self, dateiname):
        zeilen = self.datei_lesen(dateiname)

        # Dateiendung überprüfen und entsprechende Parsing-Funktion verwenden
        if dateiname.lower().endswith('.trk'):
            return self.trk_daten_parsen(zeilen)
        elif dateiname.lower().endswith('.csv'):
            return self.csv_daten_parsen(zeilen)
        else:
            print(f"Nicht unterstütztes Dateiformat für {dateiname} (muss .trk oder .csv sein).")
            return np.array([])


class KubischeBezier:
    '''
    Klasse für die Formeln der kubischen Bézierkurven (R^3).

    Manuelle Berechnung der kubischen Bézierkurve für t und die (manuell) berechneten Ableitungen.

    TODO: Eventuell mittels SymPy zwecks Vereinfachung refaktorisieren.
    '''

    def __init__(self, x0, x1, x2, x3):
        """
        Initialisierung die kubische Bézierkurve (mit gegebenen Kontrollpunkten).

        x0, x3: Stützstellen
        x1, x2: Berechnete Kontrollpunkte ai, bi
        """

        self.x0 = np.array(x0)
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.x3 = np.array(x3)

    def kubische_bezier(self, t):
        """
        Berechnet den Wert der kubischen Bézierkurve zum Parameter t.
        """

        # Helper um sicherzustellen, dass t für spätere Skalare richtig konvertiert wird (Fehlermeldungen beseitigen)
        if np.isscalar(t):
            # t als eindimensionales Numpy-Array konvertieren
            t = np.array([t])
        else:
            # Wenn t (bereits) Liste/Array ist, in normales Numpy-Array konvertieren
            t = np.array(t)

        # Erweitert t um neuen Achsenindex zwecks Vektoroperationen auf die Bézierkurven-Formelfunktion
        t = t[:, np.newaxis]

        kub_bezier = ((1 - t) ** 3 * self.x0 +
                      3 * (1 - t) ** 2 * t * self.x1 +
                      3 * (1 - t) * t ** 2 * self.x2 +
                      t ** 3 * self.x3)

        return kub_bezier

    def bezier_ableitungen(self, t):
        """
        Erste, zweite und dritte Ableitung der kubischen Bézierkurve (vektorisiert).
        """

        x0, y0, z0 = self.x0
        x1, y1, z1 = self.x1
        x2, y2, z2 = self.x2
        x3, y3, z3 = self.x3

        # Erste Ableitung (Tangentenvektor der Kurve) für x, y und z bei t
        # Zeigt in Richtung der Kurve an diesem Punkt (Geschwindigkeit bei Bewegung)
        bez_abl1_x = (-3 * x0 * (1 - t) ** 2 + 3 * x1 * (1 - t) ** 2 - 6 * x1 * (1 - t) * t + 6 * x2 * (1 - t) * t
                      - 3 * x2 * t ** 2 + 3 * x3 * t ** 2)
        bez_abl1_y = (-3 * y0 * (1 - t) ** 2 + 3 * y1 * (1 - t) ** 2 - 6 * y1 * (1 - t) * t + 6 * y2 * (1 - t) * t
                      - 3 * y2 * t ** 2 + 3 * y3 * t ** 2)
        bez_abl1_z = (-3 * z0 * (1 - t) ** 2 + 3 * z1 * (1 - t) ** 2 - 6 * z1 * (1 - t) * t + 6 * z2 * (1 - t) * t
                      - 3 * z2 * t ** 2 + 3 * z3 * t ** 2)
        bez_abl_1 = np.array([bez_abl1_x, bez_abl1_y, bez_abl1_z])

        # Zweite Ableitung (Beschleunigungsvektor der Kurve; Krümmungsvektor) für x, y und z zu t
        # Zeigt die Änderungsrate des Tangentenvektors an diesem Punkt (Beschleunigung bei Bewegung entlang der Kurve)
        bez_abl2_x = (6 * x0 * (1 - t) - 12 * x1 * (1 - t) + 6 * x2 * (1 - t) + 6 * x1 * t - 12 * x2 * t + 6 * x3 * t)
        bez_abl2_y = (6 * y0 * (1 - t) - 12 * y1 * (1 - t) + 6 * y2 * (1 - t) + 6 * y1 * t - 12 * y2 * t + 6 * y3 * t)
        bez_abl2_z = (6 * z0 * (1 - t) - 12 * z1 * (1 - t) + 6 * z2 * (1 - t) + 6 * z1 * t - 12 * z2 * t + 6 * z3 * t)
        bez_abl_2 = np.array([bez_abl2_x, bez_abl2_y, bez_abl2_z])

        # Dritte Ableitung (Änderungsrate des Krümmungsvektors (der Beschleunigung); Ruck) für x, y und z zu t
        bez_abl3_x = -6 * x0 + 18 * x1 - 18 * x2 + 6 * x3
        bez_abl3_y = -6 * y0 + 18 * y1 - 18 * y2 + 6 * y3
        bez_abl3_z = -6 * z0 + 18 * z1 - 18 * z2 + 6 * z3
        bez_abl_3 = np.array([bez_abl3_x, bez_abl3_y, bez_abl3_z])

        return bez_abl_1, bez_abl_2, bez_abl_3


class BezierKurveBerechnung:
    """
    Klasse für die Berechnung der Kontrollpunkte x1 und x2, dem Aufstellen des linearen Gleichungssystem (LGS),
    Anzeige der Matrix A und dem Vektor rhs und die der Stütz- und Kontrollpunkte und Berechnung der Bézierkurven-Punkte.
    """

    def __init__(self, stuetzpunkte):
        # Stützpunkte in ein Numpy-Array schreiben
        self.stuetzpunkte = np.asarray(stuetzpunkte)
        # Kontrollpunkte als None definieren(werden später berechnet und gespeichert)
        self._kontrollpunkte = None

    def initialisiere_lgs(self):
        """
        Initialisiert das lineare Gleichungssystem für die Berechnung der Kontrollpunkte der Bézierkurve.

        Die Gleichungen des LGS in der Funktion _setze_lgs zwecks A x = rhs Form entsprechend umgeformt worden.

        Rückgabe:
            A: Die Koeffizientenmatrix vom LGS
            rhs: Rechte Seite vom LGS
        """
        # Anzahl der Segmente; Subtraktion mit -1, da ein Segment durch zwei Stützpunkte definiert ist
        # (Anzahl der Segmente beträgt immer 1 geringer als die Anzahl der Stützpunkte)
        n = len(self.stuetzpunkte) - 1

        # Koeffizientenmatrix A des LGS mit Nullen füllen
        A = np.zeros((2 * n, 2 * n))

        # Rechte Seite (rhs) des LGS mit Nullen füllen
        rhs = np.zeros((2 * n, 3))

        return n, A, rhs

    def _setze_lgs(self, n, A, rhs):
        """
        Die Funktion berechnet die Kontrollpunkte der Bézierkurve anhand einer Liste von Stützpunkten x_0 und x_3 und setzt die Koeffizientenmatrix A
        und den Vektor rhs zu den eingelesenen Stützpunkten und Formulierung des linearen Gleichungssystem (LGS).
        Dabei werden die Gleichungen der Stetigkeitsforderungen und Anforderungen einer geschlossenen Kurve auf Basis
        der Mathematik-Vorlesung und dem Dokument "Achterbahn-Editor/-Simulator" (S. 5-6) verwendet.

        Die Gleichungen des LGS sind zwecks A x = rhs Form entsprechend umgeformt worden.

        Zwecks besserer Übersicht im LaTeX-Format (siehe alternativ im PDF-Dokument):

        - (12): B'_{i}(1) &= B'_{i+1}(0) => 2x_{i+1} = a_{i+1} + b_{i}
        - (13): B''_{i}(1) &= B''_{i+1}(0) => a_i + 2a_{i+1} = b_{i+1} + 2b_i => a_i + 2a_{i+1} - b_{i+1} - 2b_i = 0
        - (14): B'_{n-1}(1) &= B'_{0}(0) => a_0 + b_{n-1} = 2x_0
        - (15): B''_{n-1}(1) &= B''_{0}(0) => a_{n-1} - 2b_{n-1} = -2a_0 + b_0 => a_{n-1} - 2b_{n-1} + 2a_0 - b_0 = 0

        Die Koeffizienten dieser Gleichungen wurden entsprechend ihrer Position (Index) in die Matrix A eingefügt.
        Der Vektor rhs wurde mit den bekannten Werten (rechte Seite der Gleichungen) gefüllt.
        Die Reihenfolge der Gleichungen im Gleichungssystem geht von den Stetigkeitsanforderungen zu denen der geschlossenen Kurve (absteigend).

        Die Funktion gibt die Kontrollpunkte als Liste von Paaren (a_i, b_i) zurück, wobei a_i und b_i xyz-Koordinaten für x_1 und x_2 sind.

        Parameter:
           n: Anzahl Segmente auf Basis der Stützstellen
           A: Mit nullgefüllte Matrix A (linke Seite des LGS)
           rhs: Mit nullgefüllte Vektor rhs (rechte Seite des LGS)

        Rückgabe:
           A: Matrix A
           rhs: Vektor (Rechte Seite des LGS)
        """

        # Index (idx) mit Werten 0 bis n-1 (siehe initialisiere_lgs) für Anzahl der Segmente
        # für Positionierung in Matrix A und Vektor rhs. Indizierung siehe idx-Mapping.
        idx = np.arange(n)

        # --- Stetigkeitsanforderungen ---

        # Gleichungen für Stetigkeitanforderungen
        # (12'). 2x_{i+1} = a_{i+1} + b_{i} (bereits A x = rhs)
        A[2 * idx, (2 * idx + 2) % (2 * n)] = 1  # Koeffizient für a_{i+1}
        A[2 * idx, 2 * idx + 1] = 1  # Koeffizient für b_{i}
        rhs[2 * idx] = 2 * self.stuetzpunkte[(idx + 1)]  # Rechte Seite

        # (13'). a_i + 2a_{i+1} = b_{i+1} + 2b_i
        # in Form Ax = rhs: a_i + 2a_{i+1} - b_{i+1} - 2b_i = 0
        A[2 * idx + 1, 2 * idx] = 1  # Koeffizient für a_i
        A[2 * idx + 1, (2 * idx + 2) % (2 * n)] = 2  # Koeffizient für 2a_{i+1}
        A[2 * idx + 1, 2 * idx + 1] = -2  # Koeffizient für -2b_i
        A[2 * idx + 1, (2 * idx + 3) % (2 * n)] = -1  # Koeffizient für b_{i+1}

        # --- Anforderungen für geschlossene Kurven  ---

        # Gleichungen für geschlossene Kurven
        # (14'). a_0 + b_{n-1} = 2x_0 (bereits A x = rhs)
        A[2 * n - 2, 0] = 1  # Koeffizient für a_0
        A[2 * n - 2, 2 * n - 1] = 1  # Koeffizient für b_{n-1}
        rhs[2 * n - 2] = 2 * self.stuetzpunkte[0]  # Rechte Seite

        # (15'). a_{n-1} - 2b_{n-1} = -2a_0 + b_0
        # => in Form Ax = rhs: a_{n-1} - 2b_{n-1} + 2a_0 - b_0 = 0
        A[2 * n - 1, 2 * n - 2] = 1  # Koeffizient für a_{n-1}
        A[2 * n - 1, 2 * n - 1] = -2  # Koeffizient für -2b_{n-1}
        A[2 * n - 1, 0] = 2  # Koeffizient für 2a_0
        A[2 * n - 1, 1] = -1  # Koeffizient für b_0

        # Matrix A und Vektor rhs
        return A, rhs

    def berechne_kontrollpunkte(self):
        # Beim Aufruf prüfen, ob _kontrollpunkte bereits gesetzt, um doppelte Berechnungen zu vermeiden
        # Ansonsten die (bereits) gespeicherten Kontrollpunkte zurückgeben
        if self._kontrollpunkte is not None:
            return self._kontrollpunkte, _, _

        # Initialisieren und Setzen des LGS
        n, A, rhs = self.initialisiere_lgs()
        A, rhs = self._setze_lgs(n, A, rhs)

        # Lösen des Gleichungssystems
        kontrollpunkte = np.linalg.solve(A, rhs)

        # Runden der Kontrollpunkte auf 2 Dezimalstellen
        # WICHTIG: Kann deswegen minimal andere Ergebnisse bei Torsion und Krümmung liefern.
        kontrollpunkte = np.around(kontrollpunkte, decimals=2)

        # Array umformen in 2x3 (Paar von 2 mit 3 Koordinaten xyz) für ai, bi (x1, x2)
        kontrollpunkte = kontrollpunkte.reshape(-1, 2, 3)

        # Speichern der berechneten Kontrollpunkte
        self._kontrollpunkte = kontrollpunkte.tolist()

        # Ausgabe von Vektor x der 2 Unbekannten Kontrollpunkte ai und bi (_kontrollpunkte) und der Matrix A und dem Vektor rhs
        return self._kontrollpunkte, A, rhs

    @staticmethod
    def zeige_matrixA_und_vektorRHS(A, rhs):
        """
        Gibt die Matrix A und den Vektor rhs (textuell) aus.
        """
        print("\nMatrix A:")
        for row in A:
            print(' '.join(map(str, row)))

        print("\nVektor rhs:")
        for row in rhs:
            print(row)

    def ausgabe_stuetz_und_kontrollpunkte(self):
        """
        Berechnet die Kontrollpunkte basierend auf den Stützstellen und gibt alle Stütz- und Kontrollpunkte aus.
        """
        # Berechne die Kontrollpunkte
        kontrollpunkte, _, _ = self.berechne_kontrollpunkte()

        # Ausgabe der Stützpunkte
        print("\nStützpunkte:")
        for i, point in enumerate(self.stuetzpunkte):
            print(f"Punkt {i}: {point}")

        # Ausgabe der Kontrollpunkte
        print("\nKontrollpunkte:")
        for i, segment in enumerate(kontrollpunkte):
            print(f"Segment {i}:")
            print(f"a_{i}: {segment[0]}")
            print(f"b_{i}: {segment[1]}")

    def berechne_gesamte_bezierkurve(self, t_werte):
        """
        Berechnet die gesamte Bézierkurve aus den gegebenen Stütz- und Kontrollpunkten.

        Parameter:
            t_werte: Einzelne t-Werte für die Bézierkurve

        Rückgabe:
            Kurvenpunkte: Punkte auf der gesamten Bézierkurve
        """
        kontrollpunkte, _, _ = self.berechne_kontrollpunkte()

        alle_bezier_kurvenpunkte = []

        # Da geschlossene Bezierkurve mit n Stützpunkten hat, besitzt man n-1 Segmente
        for i in range(len(self.stuetzpunkte) - 1):
            # Instanz der KubischeBezier Klasse für das aktuelle Segment
            bezier_segment = KubischeBezier(self.stuetzpunkte[i], *kontrollpunkte[i],
                                            self.stuetzpunkte[(i + 1) % len(self.stuetzpunkte)])

            # Berechnet die Punkte des aktuellen Segments
            segmentpunkte = bezier_segment.kubische_bezier(t_werte)
            alle_bezier_kurvenpunkte.append(segmentpunkte)

        kurvenpunkte = np.concatenate(alle_bezier_kurvenpunkte, axis=0)

        return kurvenpunkte

    def berechne_position_fuer_globales_t(self, globales_t):
        """
        Berechnet die Position für ein gegebenes globales t (t=0 bis t=1; Position auf der gesamten Kurve) und konvertiert
        das globale t zwecks der Segmentierung (der einzelnen Bézierstücken) relativ zur Posititon der Bézierkurve.

        Ein Problem war es, akkurat die Positionierungen auf der Kurve, die aus mehreren Segmenten besteht, zu bestimmen.
        Die Idee war es, alle relevanten Information für das globale t mittels dieser Helferfunktion zu realisieren.

        ** Berechnungsgrundlage **
        - Segment-Index-Bestimmung:
          Um herauszufinden, in welchem Segment der Punkt zum globalen t liegt, wird das globale t mit der Anzahl
          der Stützpunkte multipliziert -> Skalierung auf die Anzahl der Stützpunkte.
          Es wird -1 subtrahiert, um auf die Anzahl der Segmente zu skalieren (da Anzahl n Stützstellen = Anzahl Segmente n-1).
          Durch int()-Abrunden wird der ganzzahlige Index bestimmt, in dem sich der Punkt befindet.

        - Lokales t für das Segment:
          Globales t nutzen, um die Position (lokales t) innerhalb des bestimmten Segments zu finden.
          Gleiche Skalierung wie beim Segment-Index; allerdings wird der aktuelle Index des Segments abgezogen, um
          von der Segment-Indexierung


        Parameter:
            globales_t: Das globale t (siehe oben)

        Rückgabe:
            segment_idx: Der Index des Segments
            lokales_t: Das lokale t für das Segment
            segment_kontrollpunkte: Die Kontrollpunkte des Segments
            segment_stuetzpunkte: Die Stützstellen des Segments
            position: Position auf der Bezierkurve (zum globalen t)
        """
        kontrollpunkte, _, _ = self.berechne_kontrollpunkte()

        # Index des aktuellen Segments
        segment_idx = int(globales_t * (len(self.stuetzpunkte) - 1))

        # Überprüfung des Segment-Index, dass der Segment-Index nicht gleich der Länge der berechneten kontrollpunkte ist
        # Wenn i größer oder gleich der Länge der Kontrollpunkte ist, wird i auf den letzten Index gesetzt
        if segment_idx >= len(kontrollpunkte):
            # Anpassung des Index
            segment_idx = len(kontrollpunkte) - 1

        # t-Wert innerhalb des aktuellen Segments
        lokales_t_segment = globales_t * (len(self.stuetzpunkte) - 1) - segment_idx

        # Instanz der KubischeBezier Klasse (für das aktuelle Segment)
        bezier_segment = KubischeBezier(self.stuetzpunkte[segment_idx], *kontrollpunkte[segment_idx],
                                        self.stuetzpunkte[(segment_idx + 1) % len(self.stuetzpunkte)])

        # Position auf der Bezierkurve für das lokale t (innerhalb des aktuellen Segments)
        position = bezier_segment.kubische_bezier([lokales_t_segment]).squeeze()

        # Stützstellen für das jeweilige Segment
        segment_stuetzpunkte = (
            self.stuetzpunkte[segment_idx], self.stuetzpunkte[(segment_idx + 1) % len(self.stuetzpunkte)])

        # Kontrollpunkte für das Segment ai, bi
        segment_kontrollpunkte = kontrollpunkte[segment_idx]

        return segment_idx, lokales_t_segment, segment_kontrollpunkte, segment_stuetzpunkte, position


class BezierkurveGrafiken:
    """
    Klasse für das Plotten der Grafiken der Bézierkurve.
    """

    def __init__(self, stuetzpunkte):
        self.kurve_berechnung = BezierKurveBerechnung(stuetzpunkte)
        self.stuetzpunkte = stuetzpunkte
        self.kontrollpunkte, _, _ = self.kurve_berechnung.berechne_kontrollpunkte()

        self.bz_fig = self.render_bezierkurve()
        # Helper damit der Ball nicht mehrmals gerendert wird
        self.wagon_spur_index = None

    def bz_plotten(self, kurvenpunkte):
        """
        Plotly/Dash-Konfiguration für die Darstellung der Bézierkurve.
        Wird in render_bezierkurve verwendet, um die Parameter der Kurvenpunkte zu berechnen und die gesamte Bézierkurve darzustellen.
        """
        x, y, z = kurvenpunkte[:, 0], kurvenpunkte[:, 1], kurvenpunkte[:, 2]
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='(Einzelne) Bezierkurven'))

        # Verwenden von self.kontrollpunkte und self.stuetzpunkte
        x, y, z = zip(*[kp for segment in self.kontrollpunkte for kp in segment])
        fig.add_trace(
            go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=4, color='red'), name='Kontrollpunkte'))
        x, y, z = zip(*self.stuetzpunkte)
        fig.add_trace(
            go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=4, color='green'), name='Stützpunkte'))

        fig.update_layout(
            title="Bézierkurve (mit Wagon)",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    y=1.25,
                    xanchor="center",
                    yanchor="middle",
                    buttons=[
                        dict(label="Beide anzeigen",
                             method="update",
                             args=[{"visible": [True, True, True]}]),
                        dict(label="Nur Kontrollpunkte",
                             method="update",
                             args=[{"visible": [True, True, False]}]),
                        dict(label="Nur Stützpunkte",
                             method="update",
                             args=[{"visible": [True, False, True]}]),
                        dict(label="Keine Punkte (nur Kurve)",
                             method="update",
                             args=[{"visible": [True, False, False]}]),
                    ],
                )
            ]
        )
        return fig

    def render_bezierkurve(self):
        """
        Plotten der Bézierkurve durch die Bézier-Kurvenpunkte.
        """
        t_werte = np.linspace(0, 1, 10)
        # Berechnen der Bézierkurve-Punkte
        kurvenpunkte = self.kurve_berechnung.berechne_gesamte_bezierkurve(t_werte)

        # Plotten der Bézierkurve mit den berechneten Kurvenpunkten
        return self.bz_plotten(kurvenpunkte)

    def plotte_wagon_bei_t(self, globales_t):
        """
        Plotte ein Wagon zum globalen t auf der Bézierkurve.
        """
        segment_index, lokales_t, segment_kontrollpunkte, segment_stuetzpunkte, position = self.kurve_berechnung.berechne_position_fuer_globales_t(
            globales_t)

        # Ausgabe der berechneten Informationen (in Konsole)
        # print(f"Segment Index: {segment_index}, Lokales t: {lokales_t:.2f}")
        # print(f"Segment Stützpunkte: {segment_stuetzpunkte}")
        # print(f"Segment Kontrollpunkte: {segment_kontrollpunkte}")
        # print(f"Position auf Bezierkurve (relativ zu t): {position}\n")

        # Füge der Kugel zur gespeicherten Grafik hinzu
        wagon_spur = go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]],
                                  mode='markers', marker=dict(size=10, color='orange'),
                                  name='Wagon relativ zum t-Wert')

        # Wenn es bereits eine Spur für den Ball gibt, entferne diesen
        if self.wagon_spur_index is not None:
            # Graphendaten in Liste konvertieren
            graphen_daten_liste = list(self.bz_fig.data)
            # Entferne Ball-Spur relativ zum Index
            graphen_daten_liste.pop(self.wagon_spur_index)
            # Aktualisieren des Graphen anhand der Liste
            self.bz_fig.data = tuple(graphen_daten_liste)

        # Füge die neue Wagon-Spur hinzu
        self.bz_fig.add_trace(wagon_spur)
        # ... und aktualisiere den Index
        self.wagon_spur_index = len(self.bz_fig.data) - 1

        return self.bz_fig

    def hole_infos_bei_t(self, globales_t):
        """
        Helperfunktion, zu einem globalen t weitere Informationen (Segment, lokales t, Position, ...) zu berechnen und auszugeben.
        """
        segment_index, lokales_t, segment_kontrollpunkte, segment_stuetzpunkte, position = self.kurve_berechnung.berechne_position_fuer_globales_t(
            globales_t)
        infos = [
            f"Globales t: {globales_t}",
            f"Segment Index: {segment_index}",
            f"Segment Stützpunkte: {', '.join(map(str, segment_stuetzpunkte))}",
            f"Segment Kontrollpunkte: {', '.join(map(str, segment_kontrollpunkte))}",
            f"Position auf Bézierkurve (relativ zu t): {', '.join(map(str, position))}"
        ]

        return infos

    def hole_bezierkurve(self):
        """
        Berechnen der xyz-Kurvenpunkte der Bézierkurve.
        """
        t_werte = np.linspace(0, 1, 1000)

        kurvenpunkte = self.kurve_berechnung.berechne_gesamte_bezierkurve(t_werte)
        x, y, z = kurvenpunkte[:, 0], kurvenpunkte[:, 1], kurvenpunkte[:, 2]

        return x, y, z


class BezierFormeln:
    """
    Klasse zur Erfassung aller notwendigen Formeln für die analytische Berechnungen der Bézierkurve.
    Die Formeln orientieren sich dabei aus den in der Vorlesung genannten Formeln oder der Literatur.

    Zwecks Übersicht werden die Formel einmal im LaTeX-Format hier aufgelistet (siehe auch dem PDF):


    ** Formeln **:

    - Krümmung: \( k(t) = \frac{||\dot{\vec{r}}(t) \times \ddot{\vec{r}}(t)||}{||\dot{\vec{r}}(t)||^3} \)

        Quelle gemäß:
        - [1] https://www.youtube.com/watch?v=-0t07Cv_kqM [Weitz]
        - [2] https://de.wikipedia.org/wiki/Frenetsche_Formeln#Frenetsche_Formeln_in_Abh%C3%A4ngigkeit_von_anderen_Parametern

    - Torsion:
        (1) Determinanten: \( \tau(t) = \frac{\text{det}(\dot{\vec{r}}(t), \ddot{\vec{r}}(t), \dddot{\vec{r}}(t))}{||\dot{\vec{r}}(t) \times \ddot{\vec{r}}(t)||^2} \)
        (2) Andere Formel: \( \tau(t) = \frac{(\dot{\vec{r}}(t) \times \ddot{\vec{r}}(t)) \cdot \dddot{\vec{r}}(t)}{||\dot{\vec{r}}(t) \times \ddot{\vec{r}}(t)||^2} \)

    Quelle gemäß:
        - [1] https://www.youtube.com/watch?v=-0t07Cv_kqM [Weitz]
        - [2] https://de.wikipedia.org/wiki/Frenetsche_Formeln#Frenetsche_Formeln_in_Abh%C3%A4ngigkeit_von_anderen_Parametern

    - Normen:
        Tempo: \( ||\dot{\vec{r}}(t)|| \)
        Geschwindigkeit: \( ||\ddot{\vec{r}}(t)|| \)
        Ruck: \( ||\dddot{\vec{r}}(t)|| \)

    - Frenet-Serret (Dreibein):
        T (Tangentenvektor): \vec{T}(t) = \frac{\dot{\vec{r}}(t)}{||\dot{\vec{r}}(t)||}
        B (Binormalenvektor): \vec{B}(t) = \frac{\dot{\vec{r}}(t) \times \ddot{\vec{r}}(t)}{||\dot{\vec{r}}(t) \times \ddot{\vec{r}}(t)||}
        N (Normalenvektor): \vec{N}(t) = \vec{B}(t) \times \vec{T}(t)

    Quelle gemäß:
        - [1] https://www.youtube.com/watch?v=l7eDxflL-e0
        - [2] https://de.wikipedia.org/wiki/Frenetsche_Formeln#Frenetsche_Formeln_in_Abh%C3%A4ngigkeit_von_anderen_Parametern

    - Parallelrahmen (Bishop-Rahmen):
        T (Tangentenvektor): \( \vec{T}(t) = \frac{\dot{\vec{r}}(t)}{||\dot{\vec{r}}(t)||} \)
        N und B: siehe StackExchange oder im Code (Winkelberechnung

    Quelle gemäß:
        - [1] https://math.stackexchange.com/questions/4697812/is-there-a-simple-systematic-way-to-build-a-bishop-frame
    """

    def __init__(self, bezierkurve):
        self.bezier = bezierkurve

    def kruemmung(self, t):
        # Ableitungen holen zu t
        bez1_abl_t, bez2_abl_t, _ = self.bezier.bezier_ableitungen(t)

        # Berechnung des Kreuzprodukts
        kreuz_prod = np.cross(bez1_abl_t, bez2_abl_t)

        # Berechnung der Normen
        norm_kreuz_prod = np.linalg.norm(kreuz_prod)
        norm_bez1_abl_t = np.linalg.norm(bez1_abl_t)

        # Berechnung der Krümmung
        kappa = norm_kreuz_prod / (norm_bez1_abl_t ** 3)

        return kappa

    def torsion_det_formel(self, t):
        """
        bez1_abl_t, bez2_abl_t, bez3_abl_t = self.bezier.bezier_ableitungen(t)

        # Berechnung des Determinanten der Ableitungen
        det_wert = np.linalg.det(np.vstack((bez1_abl_t, bez2_abl_t, bez3_abl_t)))

        # Berechnung des Kreuzprodukts von bez1_abl_t und bez2_abl_t
        kreuz_prod = np.cross(bez1_abl_t, bez2_abl_t)

        # Betrag des Kreuzprodukts von bez1_abl_t und bez2_abl_t
        kreuz_prod_betrag = np.linalg.norm(kreuz_prod)

        # Überprüfen, ob der Nenner fast null ist
        if kreuz_prod_betrag < 1e-10:
            return 0.0

        # Berechnung der Torsion
        tau = det_wert / (kreuz_prod_betrag ** 2)

        return tau
        """

        # Alternative Torsion-Formel zwecks Vergleich zur Determinanten-Formel:

        # Ableitungen holen zu t
        bez1_abl_t, bez2_abl_t, bez3_abl_t = self.bezier.bezier_ableitungen(t)

        # Berechnung des Kreuzprodukts von bez1_abl_t und bez2_abl_t
        kreuz_prod = np.cross(bez1_abl_t, bez2_abl_t)

        # Betrag des Kreuzprodukts von bez1_abl_t und bez2_abl_t
        kreuz_prod_betrag = np.linalg.norm(kreuz_prod)

        # Berechnung des Skalarprodukts von kreuz_prod und bez3_abl_t
        skalar_prod = np.dot(kreuz_prod, bez3_abl_t)

        # Berechnung der Torsion nach der gegebenen Formel
        tau = skalar_prod / (kreuz_prod_betrag ** 2)

        return tau

    def normen_der_ableitungen(self, t):
        # Ableitungen holen zu t
        bez1_abl_t, bez2_abl_t, bez3_abl_t = self.bezier.bezier_ableitungen(t)

        # Berechnung der Normen

        # Geschwindigkeit
        norm_bez1_abl_t = np.linalg.norm(bez1_abl_t)

        # Tempo
        norm_bez2_abl_t = np.linalg.norm(bez2_abl_t)

        # Ruck
        norm_bez3_abl_t = np.linalg.norm(bez3_abl_t)

        return norm_bez1_abl_t, norm_bez2_abl_t, norm_bez3_abl_t

    def frenet_serret(self, t):
        '''
        # Ausprobiert Formel nach englischen Wikipedia (https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas#Other_expressions_of_the_frame)
        # Dort findet eine andere Reihenfolge statt
        # Liefert dasselbe Ergebnis (!) wie die untere Implementierung; wenn zunächst erst B berechnet und dann N als Kreuzprodukt von T und B berechnet wird

        # Ableitungen holen zu t
        bez1_abl_t, bez2_abl_t, bez3_abl_t = self.bezier.bezier_ableitungen(t)

        # Tangentenvektor (T)
        T = bez1_abl_t / np.linalg.norm(bez1_abl_t)

        # Normalenvektor (N)
        # Orthogonal zum Tangentenvektor
        kreuz_prod = np.cross(bez2_abl_t, bez1_abl_t)
        N = np.cross(bez1_abl_t, kreuz_prod) / (np.linalg.norm(bez1_abl_t) * np.linalg.norm(kreuz_prod))

        # Binormalenvektor (B)
        # Orthogonal zu T und N
        B = np.cross(T, N)

        return T, N, B
        '''

        # WICHTIG! Liefert denselben orthogonalen Rahmen wie die obige Formel!
        # Formel nach Quelle im Klassen-Header

        # Ableitungen holen zu t
        bez1_abl_t, bez2_abl_t, bez3_abl_t = self.bezier.bezier_ableitungen(t)

        # Tangentenvektor (T)
        T = bez1_abl_t / np.linalg.norm(bez1_abl_t)

        # Binormalenvektor (B)
        kreuz_prod = np.cross(bez1_abl_t, bez2_abl_t)
        B = kreuz_prod / np.linalg.norm(kreuz_prod)

        # Normalenvektor (N)
        # Orthogonal zum Tangentenvektor und Binormalenvektor
        N = np.cross(B, T)

        return T, N, B

    def paralell_rahmen_bishop(self, t):
        """
        Berechnet den Parallelrahmen (auch Bishop-Frame) genannt.

        Implementiert gemäß:
        https://math.stackexchange.com/questions/4697812/is-there-a-simple-systematic-way-to-build-a-bishop-frame
        von der Antwort von Ted Shifrin

        Im Gegensatz zum Frenet-Serret (Dreibein) wird hier Wert auf Paralleltransport gelegt,
        dh. die Torsion wird weitesgehend ignoriert und resultiert nicht in 'komischen' Drehungen innerhalb der Bewegung
        entlang der Bézierkurve sofern eine hohe Torsion auftaucht.
        """

        # Ableitungen holen zu t
        bez1_abl_t, bez2_abl_t, _ = self.bezier.bezier_ableitungen(t)

        # Tangentenvektor (zeigt in die Richtung relativ zu t)
        T = bez1_abl_t / np.linalg.norm(bez1_abl_t)

        # Einheitsvektor definieren (zeigt in Z-Richtung)
        einheitsvektor = np.array([0, 0, 1])

        # Berechnung des Sinus des Winkels alpha zwischen dem Tangentenvektor und dem Einheitsvektor
        sinus_alpha = np.linalg.norm(np.cross(T, einheitsvektor))

        # Berechnung der Vektoren nu1 und nu2 (beide orthogonal zum Tangentenvektor)
        nu_1 = np.cross(T, einheitsvektor) / sinus_alpha
        nu_2 = np.cross(T, nu_1) / sinus_alpha

        # Winkelberechnen der beiden nu-Vektoren
        theta = np.arctan2(np.dot(nu_2, einheitsvektor), np.dot(nu_1, einheitsvektor))

        # Berechnung der Vektoren N und B die den orthonormalen Rahmen mit T bilden
        N = np.cos(theta) * nu_1 + np.sin(theta) * nu_2
        B = np.cross(T, N)

        # Normierung (Länge 1)
        N /= np.linalg.norm(N)
        B /= np.linalg.norm(B)

        return T, N, B


class BezierAnalyse:
    """
    Analyse-Klasse die entkoppelt zu den obigen Berechnungsformeln ist.

    Innerhalb dieser Klasse werden alle Methoden zwecks unterschiedlichen Berechnungen definiert, welche auf die
    obigen Formeln basieren und die Berechnungen relativ zum globalen t zurückgeben.
    """

    def __init__(self, stuetzpunkte):
        self.stuetzpunkte = stuetzpunkte
        self.bezier_kurve_berechnung = BezierKurveBerechnung(stuetzpunkte)

    def berechne_fuer_globales_t(self, globales_t, funktion):
        segment_index, lokales_t, _, segment_stuetzpunkte, position = self.bezier_kurve_berechnung.berechne_position_fuer_globales_t(
            globales_t)

        # Verwendendung der Kontrollpunkte direkt aus bezier_kurve_berechnung
        segment_kontrollpunkte = self.bezier_kurve_berechnung._kontrollpunkte[segment_index]

        bezier_segment = KubischeBezier(segment_stuetzpunkte[0], *segment_kontrollpunkte, segment_stuetzpunkte[1])
        bezier_formeln = BezierFormeln(bezier_segment)

        wert = funktion(bezier_formeln, lokales_t)

        return segment_index, lokales_t, position, wert

    # Funktionen um Berechnung für bestimmtes globales t zu bestimmen
    def torsion_fuer_globales_t(self, globales_t):
        return self.berechne_fuer_globales_t(globales_t, BezierFormeln.torsion_det_formel)

    def kruemmung_fuer_globales_t(self, globales_t):
        return self.berechne_fuer_globales_t(globales_t, BezierFormeln.kruemmung)

    def normen_fuer_globales_t(self, globales_t):
        return self.berechne_fuer_globales_t(globales_t, BezierFormeln.normen_der_ableitungen)

    # Funktionen um Berechnungen für alle Punkte auf der Bezierkurve zu bestimmen
    def berechne_alle_werte(self, funktion, anzahl_punkte=AUFLOESUNG_GRAFIK):
        t_werte = np.linspace(0, 1, anzahl_punkte)
        werte = []

        for t in t_werte:
            wert = self.berechne_fuer_globales_t(t, funktion)
            werte.append(wert)

        return werte

    def berechne_alle_torsionswerte(self, anzahl_punkte=AUFLOESUNG_GRAFIK):
        return self.berechne_alle_werte(BezierFormeln.torsion_det_formel, anzahl_punkte)

    def berechne_alle_kruemmungswerte(self, anzahl_punkte=AUFLOESUNG_GRAFIK):
        return self.berechne_alle_werte(BezierFormeln.kruemmung, anzahl_punkte)

    def berechne_alle_normen(self, anzahl_punkte=AUFLOESUNG_GRAFIK):
        return self.berechne_alle_werte(BezierFormeln.normen_der_ableitungen, anzahl_punkte)

    def frenet_serret(self, globales_t):
        segment_index, lokales_t, _, segment_stuetzpunkte, _ = self.bezier_kurve_berechnung.berechne_position_fuer_globales_t(
            globales_t)

        # Verwendung der Kontrollpunkte (direkt aus bezier_kurve_berechnung)
        segment_kontrollpunkte = self.bezier_kurve_berechnung._kontrollpunkte[segment_index]

        bezier_segment = KubischeBezier(segment_stuetzpunkte[0], *segment_kontrollpunkte, segment_stuetzpunkte[1])
        bezier_formeln = BezierFormeln(bezier_segment)

        return bezier_formeln.frenet_serret(lokales_t)

    def paralell_rahmen_bishop(self, globales_t):
        segment_index, lokales_t, _, segment_stuetzpunkte, _ = self.bezier_kurve_berechnung.berechne_position_fuer_globales_t(
            globales_t)

        # Verwendung der Kontrollpunkte direkt aus bezier_kurve_berechnung
        segment_kontrollpunkte = self.bezier_kurve_berechnung._kontrollpunkte[segment_index]

        bezier_segment = KubischeBezier(segment_stuetzpunkte[0], *segment_kontrollpunkte, segment_stuetzpunkte[1])
        bezier_formeln = BezierFormeln(bezier_segment)

        return bezier_formeln.paralell_rahmen_bishop(lokales_t)


class BezierVisualisierung:
    """
    Klasse für die Plotly-Visualisierung zwecks der Kurvenanalyse.

    Analyse-Grafiken:
        - Torsion zu t
        - Krümmung zu t
        - Normen zu t (Tempo, Geschwindigkeit, Ruck)
    """

    def __init__(self, bezier_analyse):
        self.bezier_analyse = bezier_analyse

    def plotte_torsion(self, t_wert_global, anzahl_punkte=AUFLOESUNG_GRAFIK):
        werte = self.bezier_analyse.berechne_alle_torsionswerte(anzahl_punkte)

        # Wert [3]: Ist der vierte Return von berechne_alle_normen (Wert der Torsion)
        torsionswerte = [wert[3] for wert in werte]
        t_werte = np.linspace(0, 1, anzahl_punkte)

        berechneter_wert = self.bezier_analyse.torsion_fuer_globales_t(t_wert_global)[3]

        fig = go.Figure(data=go.Scatter(x=t_werte, y=torsionswerte, mode='lines', line=dict(color='blue')))
        fig.add_shape(
            type="line",
            x0=t_wert_global,
            x1=t_wert_global,
            y0=min(torsionswerte),
            y1=max(torsionswerte),
            line=dict(color="green", width=2)
        )
        fig.update_layout(
            title=f"Torsion der gesamten Bézierkurve (t = {t_wert_global:.2f})<br>Berechneter Wert der Torsion an Stelle t = {t_wert_global:.2f} ist {berechneter_wert:.2f}",
            xaxis=dict(title="Globales t", tickvals=list(np.arange(0, 1.1, 0.1))),
            yaxis_title="Torsion τ(t)"
        )

        return fig

    def plotte_kruemmung(self, t_wert_global, anzahl_punkte=AUFLOESUNG_GRAFIK):
        werte = self.bezier_analyse.berechne_alle_kruemmungswerte(anzahl_punkte)

        # Wert [3]: Ist der vierte Return von berechne_alle_normen (Wert der Krümmung)
        kruemmungswerte = [wert[3] for wert in werte]
        t_werte = np.linspace(0, 1, anzahl_punkte)

        berechneter_wert = self.bezier_analyse.kruemmung_fuer_globales_t(t_wert_global)[3]

        fig = go.Figure(data=go.Scatter(x=t_werte, y=kruemmungswerte, mode='lines', line=dict(color='red')))
        fig.add_shape(
            type="line",
            x0=t_wert_global,
            x1=t_wert_global,
            y0=min(kruemmungswerte),
            y1=max(kruemmungswerte),
            line=dict(color="green", width=2)
        )
        fig.update_layout(
            title=f"Krümmung der gesamten Bézierkurve (t = {t_wert_global:.2f})<br>Berechneter Wert der Krümmung an Stelle t = {t_wert_global:.2f} ist {berechneter_wert:.2f}",
            xaxis=dict(title="Globales t", tickvals=list(np.arange(0, 1.1, 0.1))),
            yaxis_title="Krümmung κ(t)"
        )

        return fig

    def plotte_normen(self, t_wert_global, anzahl_punkte=AUFLOESUNG_GRAFIK):
        werte = self.bezier_analyse.berechne_alle_normen(anzahl_punkte)

        t_werte = np.linspace(0, 1, anzahl_punkte)

        # # Wert [3]: Ist der vierte Return von berechne_alle_normen (Wert der Torsion); zweiter Index die Positionierung
        normen_B1_t = [wert[3][0] for wert in werte]
        normen_B2_t = [wert[3][1] for wert in werte]
        normen_B3_t = [wert[3][2] for wert in werte]

        berechneter_wert = self.bezier_analyse.normen_fuer_globales_t(t_wert_global)[3]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_werte, y=normen_B1_t, mode='lines', name='Tempo'))
        fig.add_trace(go.Scatter(x=t_werte, y=normen_B2_t, mode='lines', name='Geschwindigkeit'))
        fig.add_trace(go.Scatter(x=t_werte, y=normen_B3_t, mode='lines', name='Ruck'))
        fig.add_shape(
            type="line",
            x0=t_wert_global,
            x1=t_wert_global,
            y0=0,
            y1=max(max(normen_B1_t), max(normen_B2_t), max(normen_B3_t)),
            line=dict(color="green", width=2)
        )
        fig.update_layout(
            title=f"Normen der Ableitungen der gesamten Bézierkurve (t = {t_wert_global:.2f})<br>"
                  f"Berechneter Wert der Normen an Stelle t = {t_wert_global:.2f} ist für "
                  f"Tempo: {berechneter_wert[0]:.2f}, Geschwindigkeit: {berechneter_wert[1]:.2f} und Ruck: {berechneter_wert[2]:.2f}",
            xaxis=dict(title="Globales t", tickvals=list(np.arange(0, 1.1, 0.1))),
            yaxis_title="Normen der Beträge"
        )

        return fig


class FrenetDreibeinVisualisierung:
    """
    Klasse für die Plotly-Visualisierung für Frenet-Serret (Dreibein).
    """

    def __init__(self, bezier_analyse, bezier_grafiken):
        self.bezier_analyse = bezier_analyse
        self.bezier_grafiken = bezier_grafiken
        self.vektoren_groesse = 1.0
        self.alle_positionen, self.alle_T_vektoren, self.alle_N_vektoren, self.alle_B_vektoren = self.berechne_positions_vektoren()
        self.bezier_kurve_berechnung = bezier_kurve_berechnung

    def berechne_positions_vektoren(self):
        """
        Berechnet alle Positionen und TNB-Vektoren für alle t-Werte von 0 bis 1.

        Rückgabe:
            achterbahn_positionen: Array mit den Positionen der Kugel
            T_vektoren: Array mit den T-Vektoren
            N_vektoren: Array mit den N-Vektoren
            B_vektoren: Array mit den B-Vektoren
        """

        t_werte = np.linspace(0, 1, 100)

        achterbahn_positionen = np.zeros((len(t_werte), 3))
        T_vektoren = np.zeros((len(t_werte), 3))
        N_vektoren = np.zeros((len(t_werte), 3))
        B_vektoren = np.zeros((len(t_werte), 3))

        for idx, t in enumerate(t_werte):
            # Berechnungen
            _, _, position, _ = self.bezier_analyse.berechne_fuer_globales_t(t, BezierFormeln.frenet_serret)
            T, N, B = self.bezier_analyse.frenet_serret(t)

            achterbahn_positionen[idx] = position
            T_vektoren[idx] = T
            N_vektoren[idx] = N
            B_vektoren[idx] = B

        return achterbahn_positionen, T_vektoren, N_vektoren, B_vektoren

    def plotte_wagon_und_tnb_statisch(self, t_wert_global):
        # Position und TNB-Vektoren für gegebenes t berechnen
        _, _, position, _ = self.bezier_analyse.berechne_fuer_globales_t(t_wert_global, BezierFormeln.frenet_serret)
        T, N, B = self.bezier_analyse.frenet_serret(t_wert_global)

        # Skalierung der TNB Vektoren
        T *= self.vektoren_groesse
        N *= self.vektoren_groesse
        B *= self.vektoren_groesse

        # Bezierkurve rendern
        x_kurve, y_kurve, z_kurve = self.bezier_grafiken.hole_bezierkurve()

        fig = go.Figure()

        # Bezierkurve
        fig.add_trace(go.Scatter3d(x=x_kurve, y=y_kurve, z=z_kurve, mode='lines', line=dict(color='blue', width=2),
                                   name='Bezierkurve'))

        # Wagon
        fig.add_trace(go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]], mode='markers',
                                   marker=dict(size=7, color='orange'), name='Wagon relativ zum t-Wert'))

        # TNB-Vektoren
        for vector, color, name in [(T, 'red', 'Tangentenvektor'), (N, 'green', 'Normalvektor'),
                                    (B, 'blue', 'Binormalvektor')]:
            fig.add_trace(go.Scatter3d(
                x=[position[0], position[0] + vector[0]],
                y=[position[1], position[1] + vector[1]],
                z=[position[2], position[2] + vector[2]],
                mode='lines',
                line=dict(color=color, width=6),
                name=name
            ))

        fig.update_layout(title=f"Frenet-Serret (Dreibein) mit den T, N und B Vektoren bei t = {t_wert_global:.2f}")

        fig.show()

    # Für die Animation (ähnlich wie oben; nur zwecks Performanz refaktorisiert)
    def initialisiere_grafik(self):
        """Initialisiert die Grafik und gibt die aktualisiert Grafik zurück."""
        x_kurven_wert, y_kurven_wert, z_kurven_wert = self.bezier_grafiken.hole_bezierkurve()

        fig = go.Figure()

        # Bezierkurve
        fig.add_trace(go.Scatter3d(x=x_kurven_wert, y=y_kurven_wert, z=z_kurven_wert, mode='lines',
                                   line=dict(color='blue', width=2), name='Bézierkurve'))

        # Platzhalter für Wagon
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                                   marker=dict(size=7, color='orange'), name='Wagon relativ zum t-Wert'))

        # Platzhalter für TNB-Vektoren
        for color, name in [('red', 'Tangentenvektor'), ('green', 'Normalvektor'), ('blue', 'Binormalvektor')]:
            fig.add_trace(go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[0, 0],
                mode='lines',
                line=dict(color=color, width=6),
                name=name
            ))

        return fig

    def aktualisiere_grafik(self, fig, t_wert_global, zeige_details=True):
        # Da man 100 t-Werte hat für den t-Slider
        idx = int(min(t_wert_global, 0.99) * 100)

        position = self.alle_positionen[idx]
        T = self.alle_T_vektoren[idx] * self.vektoren_groesse
        N = self.alle_N_vektoren[idx] * self.vektoren_groesse
        B = self.alle_B_vektoren[idx] * self.vektoren_groesse

        # Wagon
        fig.data[1].x = [position[0]]
        fig.data[1].y = [position[1]]
        fig.data[1].z = [position[2]]

        # TNB-Vektoren aktualisieren
        for i, vector in enumerate([T, N, B]):
            fig.data[i + 2].x = [position[0], position[0] + vector[0]]
            fig.data[i + 2].y = [position[1], position[1] + vector[1]]
            fig.data[i + 2].z = [position[2], position[2] + vector[2]]

        fig.update_layout(
            title=f"Frenet-Serret (Dreibein) mit den T, N und B Vektoren bei t = {t_wert_global:.2f}",
            scene=dict(
                xaxis=dict(showbackground=zeige_details, title_text="x" if zeige_details else "",
                           showgrid=zeige_details, zeroline=zeige_details, showticklabels=zeige_details),
                yaxis=dict(showbackground=zeige_details, title_text="y" if zeige_details else "",
                           showgrid=zeige_details, zeroline=zeige_details, showticklabels=zeige_details),
                zaxis=dict(showbackground=zeige_details, title_text="z" if zeige_details else "",
                           showgrid=zeige_details, zeroline=zeige_details, showticklabels=zeige_details)
            )
        )

        return fig

    def hole_infos_und_frenet_vektoren_bei_t(self, t_wert_global):

        # Hole Grundinfos
        segment_index, lokales_t, segment_kontrollpunkte, segment_stuetzpunkte, position = \
            self.bezier_kurve_berechnung.berechne_position_fuer_globales_t(t_wert_global)

        # Hole Dreibein TNB-Vektoren
        T, N, B = self.bezier_analyse.frenet_serret(t_wert_global)

        # Infos
        infos = {
            'Globales t': t_wert_global,
            'Bézierkurve-Segment Index': segment_index,
            'Lokales t': lokales_t,
            'Position auf der Bézierkurve': position,
            'Segment Stützpunkte': segment_stuetzpunkte,
            'Segment Kontrollpunkte': segment_kontrollpunkte,
            'Tangentenvektor (T)': T,
            'Normalenvektor (N)': N,
            'Binormalenvektor (B)': B
        }

        return infos


class ParallelRahmenVisualisierung:
    """
    Klasse für die Plotly-Visualisierung für den Parallel-Rahmen (Bishop).

    Info: Selbe Struktur die die Frenet-Serret Klasse.
    TODO: Refaktorisierung (Basisklasse mit zwei Subklassen für Frenet-Serret (Dreibein) und dem Parallelrahmen)
    """

    def __init__(self, bezier_analyse, bezier_grafiken):
        self.bezier_analyse = bezier_analyse
        self.bezier_grafiken = bezier_grafiken
        self.vektoren_groesse = 1.0
        self.alle_positionen, self.alle_T_vektoren, self.alle_N_vektoren, self.alle_B_vektoren = self.berechne_positions_vektoren()
        self.bezier_kurve_berechnung = bezier_kurve_berechnung

    def berechne_positions_vektoren(self):
        t_werte = np.linspace(0, 1, 100)

        achterbahn_positionen = np.zeros((len(t_werte), 3))
        T_vektoren = np.zeros((len(t_werte), 3))
        N_vektoren = np.zeros((len(t_werte), 3))
        B_vektoren = np.zeros((len(t_werte), 3))

        for idx, t in enumerate(t_werte):
            _, _, position, _ = self.bezier_analyse.berechne_fuer_globales_t(t, BezierFormeln.paralell_rahmen_bishop)
            T, N, B = self.bezier_analyse.paralell_rahmen_bishop(t)

            achterbahn_positionen[idx] = position
            T_vektoren[idx] = T
            N_vektoren[idx] = N
            B_vektoren[idx] = B

        return achterbahn_positionen, T_vektoren, N_vektoren, B_vektoren

    def plotte_wagon_und_tnb_statisch(self, t_wert_global):
        idx = int(min(t_wert_global, 0.99) * 100)
        position = self.alle_positionen[idx]
        T = self.alle_T_vektoren[idx] * self.vektoren_groesse
        N = self.alle_N_vektoren[idx] * self.vektoren_groesse
        B = self.alle_B_vektoren[idx] * self.vektoren_groesse

        x_kurve, y_kurve, z_kurve = self.bezier_grafiken.hole_bezierkurve()

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(x=x_kurve, y=y_kurve, z=z_kurve, mode='lines', line=dict(color='blue', width=2),
                                   name='Bézierkurve'))
        fig.add_trace(go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]], mode='markers',
                                   marker=dict(size=7, color='orange'), name='Wagon relativ zum t-Wert'))

        for vector, color, name in [(T, 'red', 'Tangentenvektor'), (N, 'green', 'Normalenvektor'),
                                    (B, 'blue', 'Binormalvektor')]:
            fig.add_trace(
                go.Scatter3d(x=[position[0], position[0] + vector[0]], y=[position[1], position[1] + vector[1]],
                             z=[position[2], position[2] + vector[2]], mode='lines', line=dict(color=color, width=6),
                             name=name))

        fig.update_layout(title=f"Parallel-Rahmen (Bishop) mit den T, N und B Vektoren bei t = {t_wert_global:.2f}")

        # In return fig ändern, wenn in Dash statt Plotly
        fig.show()

    def initialisiere_grafik(self):
        x_kurven_wert, y_kurven_wert, z_kurven_wert = self.bezier_grafiken.hole_bezierkurve()

        fig = go.Figure()

        # Bezierkurve
        fig.add_trace(go.Scatter3d(x=x_kurven_wert, y=y_kurven_wert, z=z_kurven_wert, mode='lines',
                                   line=dict(color='blue', width=2), name='Bezierkurve'))

        # Platzhalter für Wagon
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                                   marker=dict(size=7, color='orange'), name='Wagon relativ zum t-Wert'))

        # Platzhalter für TNB-Vektoren
        for color, name in [('red', 'Tangentenvektor'), ('green', 'Normalenvektor'), ('blue', 'Binormalvektor')]:
            fig.add_trace(go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[0, 0],
                mode='lines',
                line=dict(color=color, width=6),
                name=name
            ))

        return fig

    def aktualisiere_grafik(self, fig, t_wert_global, zeige_details=True):
        # Da man 100 t-Werte hat für den t-Slider (ansonsten Fehler bei t=1)
        idx = int(min(t_wert_global, 0.99) * 100)

        position = self.alle_positionen[idx]

        T = self.alle_T_vektoren[idx] * self.vektoren_groesse
        N = self.alle_N_vektoren[idx] * self.vektoren_groesse
        B = self.alle_B_vektoren[idx] * self.vektoren_groesse

        # Wagon
        fig.data[1].x = [position[0]]
        fig.data[1].y = [position[1]]
        fig.data[1].z = [position[2]]

        # TNB-Vektoren aktualisieren
        for i, vector in enumerate([T, N, B]):
            fig.data[i + 2].x = [position[0], position[0] + vector[0]]
            fig.data[i + 2].y = [position[1], position[1] + vector[1]]
            fig.data[i + 2].z = [position[2], position[2] + vector[2]]

        fig.update_layout(
            title=f"Parallel-Rahmen (Bishop) mit den T, N und B Vektoren bei t = {t_wert_global:.2f}",
            scene=dict(
                xaxis=dict(showbackground=zeige_details, title_text="x" if zeige_details else "",
                           showgrid=zeige_details, zeroline=zeige_details, showticklabels=zeige_details),
                yaxis=dict(showbackground=zeige_details, title_text="y" if zeige_details else "",
                           showgrid=zeige_details, zeroline=zeige_details, showticklabels=zeige_details),
                zaxis=dict(showbackground=zeige_details, title_text="z" if zeige_details else "",
                           showgrid=zeige_details, zeroline=zeige_details, showticklabels=zeige_details)
            )
        )

        return fig

    def hole_infos_und_bishop_vektoren_bei_t(self, t_wert_global):

        # Hole Grundinfos
        segment_index, lokales_t, segment_kontrollpunkte, segment_stuetzpunkte, position = \
            self.bezier_kurve_berechnung.berechne_position_fuer_globales_t(t_wert_global)

        # Hole Paralellrahmen TNB-Vektoren
        T, N, B = self.bezier_analyse.paralell_rahmen_bishop(t_wert_global)

        # Infos
        infos = {
            'Globales t': t_wert_global,
            'Bézierkurve-Segment Index': segment_index,
            'Lokales t': lokales_t,
            'Position auf der Bezierkurve': position,
            'Segment Stützpunkte': segment_stuetzpunkte,
            'Segment Kontrollpunkte': segment_kontrollpunkte,
            'Tangentenvektor (T)': T,
            'Normalenvektor (N)': N,
            'Binormalenvektor (B)': B
        }

        return infos


class TorusKnoten:
    """
    Berechnung von Torusknoten für kubische Bezierkurven unter Berücksichtigung der Skalierung  von t-Werten
    im Bereich t=0 bis t=1 auf 0 bis 2pi und realisiert die Konvertierung von Torus-Koordinaten
    in kartesische Koordinaten.
    """

    def __init__(self, R=5, r=2, p=2, q=3):
        """
        Initialisierung der Parameter (Standard: siehe Parametrisierung; R=5, r=2, p=2 und q=3)
        """
        self.R = R
        self.r = r
        self.p = p
        self.q = q

    def torus_knoten(self, t):
        """
        Berechnet die Koordinaten des Torusknotens für einen t-Wert.
        Hier skaliert t (0 bis 1) auf 0 bis 2pi.
        """
        # Mappen vom t-Wert Parameter zu 0 bis 2pi
        theta = 2 * np.pi * t

        # XY Projektion in der Ebene
        x = (self.R + self.r * np.cos(self.p * theta)) * np.cos(self.q * theta)
        y = (self.R + self.r * np.cos(self.p * theta)) * np.sin(self.q * theta)
        # Höhe der Kurve
        z = self.r * np.sin(self.p * theta)

        return x, y, z

    def speichere_knotenpunkte(self, filename="torus.csv", anzahl=100):
        """
        Berechnet die Koordinaten des Torusknotens für eine definierte Anzahl von Punkten + speichert diese in Datei.
        ACHTUNG: Mehrmals speichert resultiert in der Überschreibung von torus.txt
        """
        # Anzahl der einzelnen Punkte zwischen t=0 und t=1
        t_werte = np.linspace(0, 1, anzahl)

        with open(filename, "w") as file:
            for t in t_werte:
                x, y, z = self.torus_knoten(t)
                # Im Format x; y; z (2 Dezimalstellen)
                file.write(f"{x:.2f};{y:.2f};{z:.2f}\n")

        print(f"Torus in '{filename}' gespeichert.")


if __name__ == "__main__":

    # Stützpunkte definieren
    # (KONFIGURATION DER STRECKE IM HEADER)

    # Einlesen der Stützpunkte
    stuetzpunkte = LeseDatei().trackpunkte_einlesen(f"{UNTERORDNER_STRECKEN}/{STRECKE}")

    # Instanz BezierKurveBerechnung
    bezier_kurve_berechnung = BezierKurveBerechnung(stuetzpunkte)

    # Kontrollpunkte berechnen
    kontrollpunkte, _, _ = bezier_kurve_berechnung.berechne_kontrollpunkte()

    # ---- ---- ---- ----

    # Grafiken und Analyse initialisieren
    grafiken = BezierkurveGrafiken(stuetzpunkte)
    analyse = BezierAnalyse(stuetzpunkte)

    # Frenet-Serret-Dreibein-Visualisierung
    fs_visualisierung = FrenetDreibeinVisualisierung(analyse, grafiken)
    aktuelle_fig_frenet = fs_visualisierung.initialisiere_grafik()

    # Parallel-Rahmen-Visualisierung
    parallel_bishop_visualisierung = ParallelRahmenVisualisierung(analyse, grafiken)
    aktuelle_fig_bishop = parallel_bishop_visualisierung.initialisiere_grafik()

    # Dateiname für Darstellung (ohne Endung)
    dateiname_ohne_endung = STRECKE.split('.')[0]

    # Konfiguration von Dash
    app = dash.Dash(__name__)
    app.title = "MMCG (2023)"

    app.layout = html.Div([
        html.H1("MMCG (2023) - Demo", style={'textAlign': 'center'}),
        html.Br(),
        html.Div([
            html.Strong("Eingelesene Datei:"),
            f" {dateiname_ohne_endung}"
        ], style={'textAlign': 'center'}),
        # Wagon auf der Bezierkurve anzeigen
        html.H2("Bézierkurve mit Wagon", style={'textAlign': 'center'}),
        html.Div([
            # html.H2("Wagon auf der Bézierkurve"),
            dcc.Graph(id='wagon-graph', figure=grafiken.plotte_wagon_bei_t(t_wert_global))
        ]),
        # Globaler t-Wert-Slider und Diagrammselektion
        html.H2("t-Wert anpassen:", style={'textAlign': 'center'}),
        html.Div([
            html.Label("Globales t für Wagonposition und Analyse:", style={'textAlign': 'center'}),
            dcc.Slider(
                id='global-t-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.5,
                marks={i / 10: f"{i / 10}" for i in range(11)}
            ),
            html.H2("Krümmung, Torsion und Normen", style={'textAlign': 'center'}),
            html.Div([
                html.Label("Wählen die Grafiken zur Darstellung aus:", style={'textAlign': 'center'}),
                html.Br(),
                dcc.Checklist(
                    id='graph-checklist',
                    options=[
                        {'label': 'Krümmung', 'value': 'kruemmung'},
                        {'label': 'Torsion', 'value': 'torsion'},
                        {'label': 'Normen', 'value': 'normen'}
                    ],
                    value=['torsion', 'kruemmung', 'normen'],
                    inline=True
                )
            ], style={'textAlign': 'center'}),
        ]),
        # Analyse-Grafiken
        html.Div([
            dcc.Graph(id='kruemmung-graph'),
            dcc.Graph(id='torsion-graph'),
            dcc.Graph(id='normen-graph')
        ]),
        html.Hr(),
        # Für das Frenet-Dreibein
        html.Div([
            html.H2("Frenet-Serret (Dreibein) Rahmen", style={'textAlign': 'center'}),
            dcc.Graph(id='frenet-graph', figure=aktuelle_fig_frenet),
            html.H2("t-Wert anpassen:", style={'textAlign': 'center'}),
            dcc.Slider(
                id='frenet-slider',
                min=0,
                max=1,
                value=t_wert_frenet_parallel,
                marks={i / 100: str(i / 100) for i in range(0, 101, 5)},
                step=0.01,
                updatemode='drag'
            ),
            html.Div(id='frenet-info-text',
                     style={'textAlign': 'center', 'padding': '10px', 'background-color': '#f9f9f9',
                            'margin': '10px', 'border-radius': '5px'})
        ]),
        html.Hr(),
        # Für den Parallel-Rahmen
        html.Div([
            html.H2("Parallel-Rahmen (Bishop)", style={'textAlign': 'center'}),
            dcc.Graph(id='bishop-graph', figure=aktuelle_fig_bishop),
            html.H2("t-Wert anpassen:", style={'textAlign': 'center'}),
            dcc.Slider(
                id='bishop-slider',
                min=0,
                max=1,
                value=t_wert_frenet_parallel,
                marks={i / 100: str(i / 100) for i in range(0, 101, 5)},
                step=0.01,
                updatemode='drag'
            ),
            html.Div(id='bishop-info-text',
                     style={'textAlign': 'center', 'padding': '10px', 'background-color': '#f9f9f9',
                            'margin': '10px', 'border-radius': '5px'})
        ])
    ])


    # Callbacks (für Grafiken)
    @app.callback([
        Output('torsion-graph', 'figure'),
        Output('kruemmung-graph', 'figure'),
        Output('normen-graph', 'figure')],
        [Input('global-t-slider', 'value'),
         Input('graph-checklist', 'value')
         ])
    def update_graphs(t_wert, ausgewaehlte_diagramme):
        bezier_vis = BezierVisualisierung(BezierAnalyse(stuetzpunkte))

        torsion_fig = kruemmung_fig = normen_fig = {}
        if 'torsion' in ausgewaehlte_diagramme:
            torsion_fig = bezier_vis.plotte_torsion(t_wert)
        if 'kruemmung' in ausgewaehlte_diagramme:
            kruemmung_fig = bezier_vis.plotte_kruemmung(t_wert)
        if 'normen' in ausgewaehlte_diagramme:
            normen_fig = bezier_vis.plotte_normen(t_wert)

        return torsion_fig, kruemmung_fig, normen_fig


    # Callbacks (für Frenet-Serret Rahmen)
    @app.callback(
        [Output('frenet-graph', 'figure'),
         Output('frenet-info-text', 'children')],
        [Input('frenet-slider', 'value')]
    )
    def update_frenet(aktueller_t_wert):
        fig = fs_visualisierung.aktualisiere_grafik(aktuelle_fig_frenet, aktueller_t_wert)

        infos = fs_visualisierung.hole_infos_und_frenet_vektoren_bei_t(aktueller_t_wert)
        info_divs = [html.Div(f"{key}: {value}") for key, value in infos.items()]

        return fig, info_divs


    # Callbacks (für Parallel-Rahmen)
    @app.callback(
        [Output('bishop-graph', 'figure'),
         Output('bishop-info-text', 'children')],
        [Input('bishop-slider', 'value')]
    )
    def update_bishop(aktueller_t_wert):
        fig = parallel_bishop_visualisierung.aktualisiere_grafik(aktuelle_fig_bishop, aktueller_t_wert)

        infos = parallel_bishop_visualisierung.hole_infos_und_bishop_vektoren_bei_t(aktueller_t_wert)
        info_divs = [html.Div(f"{key}: {value}") for key, value in infos.items()]

        return fig, info_divs


    # Callbacks für den Wagon auf der Bezierkurve
    @app.callback(
        Output('wagon-graph', 'figure'),
        [Input('global-t-slider', 'value')]
    )
    def update_wagon(t_wert):
        return grafiken.plotte_wagon_bei_t(t_wert)


    # App starten
    app.run_server(debug=True)

    # --- Torusknoten generieren

    # Instanz Torusknoen
    # torus_knoten = TorusKnoten()

    # Speichere Torus
    # Muss (manuell) in den Strecken-Unterordner verschoben werden, sofern die Strecke benutzt werden soll
    # torus_knoten.speichere_knotenpunkte(filename="torus.csv", anzahl=100)
