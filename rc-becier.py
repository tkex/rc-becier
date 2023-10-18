"""
Bezierkurven
Projektbeschreibung (TODO)
"""

import numpy as np
import plotly.graph_objects as go


class LeseDatei:

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

            x, y, z = map(float, zeile.split(';'))  # Für andere Trennzeichen, diesen Wert anpassen
            stuetzpunkte.append((x, y, z))

        return np.array(stuetzpunkte)

    def trackpunkte_einlesen(self, dateiname):
        zeilen = self.datei_lesen(dateiname)

        # Dateiendung überprüfen und entsprechende korrekte Parsing-Funktion verwenden
        if dateiname.lower().endswith('.trk'):
            return self.trk_daten_parsen(zeilen)
        elif dateiname.lower().endswith('.csv'):
            return self.csv_daten_parsen(zeilen)
        else:
            print(f"Nicht unterstütztes Dateiformat für {dateiname}.")
            return np.array([])



class KubischeBezier:
    def __init__(self, x0, x1, x2, x3):
        """
        Initialisiert die kubische Bézier-Kurve mit gegebenen Kontrollpunkten.
        """

        self.x0 = np.array(x0)
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.x3 = np.array(x3)

    def kubische_bezier(self, t):
        """
        Berechnet den Wert der kubischen Bézier-Kurve bei einem gegebenen Parameter t.
        """

        if np.isscalar(t):
            t = np.array([t])
        else:
            t = np.array(t)

        t = t[:, np.newaxis]

        kub_bezier = ((1 - t) ** 3 * self.x0 +
                      3 * (1 - t) ** 2 * t * self.x1 +
                      3 * (1 - t) * t ** 2 * self.x2 +
                      t ** 3 * self.x3)

        return kub_bezier

    def bezier_ableitungen(self, t):
        """
        Erste, zweite und dritte Ableitung der kubischen Bezierkurve (manuell + vektorisiert).
        TODO: Eventuell mit SymPy refaktorisieren
        """

        x0, y0, z0 = self.x0
        x1, y1, z1 = self.x1
        x2, y2, z2 = self.x2
        x3, y3, z3 = self.x3

        # Erste Ableitung (Tangentenvektor der Kurve) für x, y und z
        bez_abl1_x = (-3 * x0 * (1 - t) ** 2 + 3 * x1 * (1 - t) ** 2 - 6 * x1 * (1 - t) * t + 6 * x2 * (1 - t) * t
                  - 3 * x2 * t ** 2 + 3 * x3 * t ** 2)
        bez_abl1_y = (-3 * y0 * (1 - t) ** 2 + 3 * y1 * (1 - t) ** 2 - 6 * y1 * (1 - t) * t + 6 * y2 * (1 - t) * t
                  - 3 * y2 * t ** 2 + 3 * y3 * t ** 2)
        bez_abl1_z = (-3 * z0 * (1 - t) ** 2 + 3 * z1 * (1 - t) ** 2 - 6 * z1 * (1 - t) * t + 6 * z2 * (1 - t) * t
                  - 3 * z2 * t ** 2 + 3 * z3 * t ** 2)
        bez_abl_1 = np.array([bez_abl1_x, bez_abl1_y, bez_abl1_z])

        # Zweite Ableitung (Normalenvektor der Kurve dh. Richtung des Tangentenvektors) für x, y und z
        bez_abl2_x = (6 * x0 * (1 - t) - 12 * x1 * (1 - t) + 6 * x2 * (1 - t) + 6 * x1 * t - 12 * x2 * t + 6 * x3 * t)
        bez_abl2_y = (6 * y0 * (1 - t) - 12 * y1 * (1 - t) + 6 * y2 * (1 - t) + 6 * y1 * t - 12 * y2 * t + 6 * y3 * t)
        bez_abl2_z = (6 * z0 * (1 - t) - 12 * z1 * (1 - t) + 6 * z2 * (1 - t) + 6 * z1 * t - 12 * z2 * t + 6 * z3 * t)
        bez_abl_2 = np.array([bez_abl2_x, bez_abl2_y, bez_abl2_z])

        # Dritte Ableitung
        bez_abl3_x = -6 * x0 + 18 * x1 - 18 * x2 + 6 * x3
        bez_abl3_y = -6 * y0 + 18 * y1 - 18 * y2 + 6 * y3
        bez_abl3_z = -6 * z0 + 18 * z1 - 18 * z2 + 6 * z3
        bez_abl_3 = np.array([bez_abl3_x, bez_abl3_y, bez_abl3_z])

        return bez_abl_1, bez_abl_2, bez_abl_3



class BezierKurveBerechnung:

    def __init__(self, stuetzpunkte):
        self.stuetzpunkte = np.asarray(stuetzpunkte)
        # Speichert die Kontrollpunkte
        self._kontrollpunkte = None

    def initialisiere_lgs(self):
        """
        Initialisiert das lineare Gleichungssystem für die Berechnung der Kontrollpunkte der Bézier-Kurve.

        Die Gleichungen des LGS sind zwecks A x = rhs Form entsprechend umgeformt worden.

        Rückgabe:
            A: Die Koeffizientenmatrix des linearen Gleichungssystems.
            rhs: Die rechte Seite des linearen Gleichungssystems.
        """
        n = len(self.stuetzpunkte) - 1
        A = np.zeros((2 * n, 2 * n))
        rhs = np.zeros((2 * n, 3))
        return n, A, rhs

    def _setze_lgs(self, n, A, rhs):
        idx = np.arange(n)

        # Gleichungen für Stetigkeitanforderungen
        # IV. 2x_{i+1} = a_{i+1} + b_{i}
        A[2 * idx, (2 * idx + 2) % (2 * n)] = 1
        A[2 * idx, 2 * idx + 1] = 1
        rhs[2 * idx] = 2 * self.stuetzpunkte[(idx + 1) % n]

        # III. a_i + 2a_{i+1} = b_{i+1} + 2b_i
        A[2 * idx + 1, 2 * idx] = 1
        A[2 * idx + 1, (2 * idx + 2) % (2 * n)] = 2
        A[2 * idx + 1, 2 * idx + 1] = -2
        A[2 * idx + 1, (2 * idx + 3) % (2 * n)] = -1

        # Gleichungen für geschlossene Kurven
        # I. a_0 + b_{n-1} = 2x_0
        A[2 * n - 2, 0] = 1
        A[2 * n - 2, 2 * n - 1] = 1
        rhs[2 * n - 2] = 2 * self.stuetzpunkte[0]

        # II. a_{n-1} - 2b_{n-1} = -2a_0 + b_0
        A[2 * n - 1, 2 * n - 2] = 1
        A[2 * n - 1, 2 * n - 1] = -2
        A[2 * n - 1, 0] = 2
        A[2 * n - 1, 1] = -1

        return A, rhs

    def berechne_kontrollpunkte(self):
        # Beim Aufruf prüfen, ob _kontrollpunkte bereits gesetzt, um doppelte Berechnungen zu vermeiden
        # Ansonsten die (bereits) gespeicherten Kontrollpunkte zurückgeben
        if self._kontrollpunkte is not None:
            return self._kontrollpunkte, _, _

        n, A, rhs = self.initialisiere_lgs()
        A, rhs = self._setze_lgs(n, A, rhs)

        # Lösen des Gleichungssystems
        kontrollpunkte = np.linalg.solve(A, rhs)

        # Runden der Kontrollpunkte auf 2 Dezimalstellen
        # WICHTIG: Kann deswegen minimal andere Ergebnisse bei Torsion und Krümmung liefern.
        kontrollpunkte = np.around(kontrollpunkte, decimals=2)

        kontrollpunkte = kontrollpunkte.reshape(-1, 2, 3)

        # Speichern der berechneten Kontrollpunkte
        self._kontrollpunkte = kontrollpunkte.tolist()

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
            t_werte (array): Die einzelnen Parameterwerte für die Bézierkurve.

        Rückgabe:
            Kurvenpunkte (Array): Die Punkte der gesamten Bézierkurve.
        """
        kontrollpunkte, _, _ = self.berechne_kontrollpunkte()

        alle_bezier_kurvenpunkte = []

        # Da geschlossene Bezierkurve mit n Stützpunkten hat, besitzt man n-1 Segmente
        for i in range(len(self.stuetzpunkte) - 1):
            # Erstellt jeweils eine Instanz der KubischeBezier Klasse für das aktuelle Segment
            bezier_segment = KubischeBezier(self.stuetzpunkte[i], *kontrollpunkte[i],
                                            self.stuetzpunkte[(i + 1) % len(self.stuetzpunkte)])

            # Berechnet die Punkte des aktuellen Segments
            segmentpunkte = bezier_segment.kubische_bezier(t_werte)
            alle_bezier_kurvenpunkte.append(segmentpunkte)

        kurvenpunkte = np.concatenate(alle_bezier_kurvenpunkte, axis=0)

        return kurvenpunkte

    def berechne_position_fuer_globales_t(self, globales_t):
        """
        Berechnet die Koordinaten für ein gegebenes globales t (t=0 bis t=1).

        Parameter:
            globales_t: Das globale t, ein Wert zwischen 0 und 1.

        Rückgabe:
            segment_index: Der Index des Segments.
            lokales_t: Das lokale t für das Segment.
            segment_kontrollpunkte: Die Kontrollpunkte des Segments.
            segment_stuetzpunkte: Die Stützstellen des Segments.
            position: Die Koordinaten auf der Bezierkurve zum gegebenen globalen t.
        """
        kontrollpunkte, _, _ = self.berechne_kontrollpunkte()

        # Index des aktuellen Segments
        i = int(globales_t * (len(self.stuetzpunkte) - 1))

        # Wenn i größer oder gleich der Länge der Kontrollpunkte ist, wird i auf den letzten gültigen Index gesetzt.
        # Überprüfung des Index
        if i >= len(kontrollpunkte):
            # Anpassung des Index
            i = len(kontrollpunkte) - 1

        # t-Wert innerhalb des aktuellen Segments
        lokales_t_segment = globales_t * (len(self.stuetzpunkte) - 1) - i

        # Erstellt eine Instanz der KubischeBezier Klasse für das aktuelle Segment
        bezier_segment = KubischeBezier(self.stuetzpunkte[i], *kontrollpunkte[i],
                                        self.stuetzpunkte[(i + 1) % len(self.stuetzpunkte)])

        # Berechne die Position auf der Bezierkurve für das lokale t innerhalb des aktuellen Segments
        position = bezier_segment.kubische_bezier([lokales_t_segment]).squeeze()

        # Hole die Stützstellen für das jeweilige Segment
        segment_stuetzpunkte = (self.stuetzpunkte[i], self.stuetzpunkte[(i + 1) % len(self.stuetzpunkte)])

        # Hole die Kontrollpunkte für das Segment ai, bi
        segment_kontrollpunkte = kontrollpunkte[i]

        return i, lokales_t_segment, segment_kontrollpunkte, segment_stuetzpunkte, position


class BezierkurveGrafiken:
    def __init__(self, stuetzpunkte):
        self.kurve_berechnung = BezierKurveBerechnung(stuetzpunkte)
        self.stuetzpunkte = stuetzpunkte
        self.kontrollpunkte, _, _ = self.kurve_berechnung.berechne_kontrollpunkte()

        self.bz_fig = self.render_bezierkurve()
        # Helper damit der Ball nicht mehrmals gerendert wird
        self.wagon_spur_index = None


    def bz_plotten(self, kurvenpunkte):
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
            width=1200,
            height=800,
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
        t_werte = np.linspace(0, 1, 10)
        kurvenpunkte = self.kurve_berechnung.berechne_gesamte_bezierkurve(t_werte)
        # Direktes Return der Grafik
        return self.bz_plotten(kurvenpunkte)

    def hole_dash_grafik(self):
        #return dcc.Graph(figure=self.bz_grafik)
        return dcc.Graph(id='bz-graph-basis', figure=self.bz_fig)

    def zeichne_wagon_bei_t(self, globales_t):
        segment_index, lokales_t, segment_kontrollpunkte, segment_stuetzpunkte, position = self.kurve_berechnung.berechne_position_fuer_globales_t(
            globales_t)

        # Ausgabe der berechneten Informationen (in Konsole)
        #print(f"Segment Index: {segment_index}, Lokales t: {lokales_t:.2f}")
        #print(f"Segment Stützpunkte: {segment_stuetzpunkte}")
        #print(f"Segment Kontrollpunkte: {segment_kontrollpunkte}")
        #print(f"Position auf Bezierkurve (relativ zu t): {position}\n")

        # Füge der Kugel zur gespeicherten Grafik hinzu
        wagon_spur = go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]],
                                  mode='markers', marker=dict(size=10, color='black'),
                                  name='Globale t-Wert Position')

        # Wenn es bereits eine Spur für den Ball gibt, entfernen
        if self.wagon_spur_index is not None:
            # Graphendaten in Liste konvertieren
            graphen_daten_liste = list(self.bz_fig.data)
            # Entferne Ball-Spur relativ zum Index
            graphen_daten_liste.pop(self.wagon_spur_index)
            # Aktualisieren des Graphens anhand der Liste
            self.bz_fig.data = tuple(graphen_daten_liste)

        # Füge die neue Wagon-Spur hinzu
        self.bz_fig.add_trace(wagon_spur)
        # ... und aktualisiere den Index
        self.wagon_spur_index = len(self.bz_fig.data) - 1

        return self.bz_fig

    def hole_infos_bei_t(self, globales_t):
        segment_index, lokales_t, segment_kontrollpunkte, segment_stuetzpunkte, position = self.kurve_berechnung.berechne_position_fuer_globales_t(
            globales_t)
        infos = [
            f"Globales t: {globales_t}",
            f"Segment Index: {segment_index}",
            f"Segment Stützpunkte: {', '.join(map(str, segment_stuetzpunkte))}",
            f"Segment Kontrollpunkte: {', '.join(map(str, segment_kontrollpunkte))}",
            f"Position auf Bezierkurve (relativ zu t): {', '.join(map(str, position))}"
        ]
        return infos

class BezierFormeln:
    def __init__(self, bezier_curve):
        self.bezier = bezier_curve

    def kruemmung(self, t):
        bez1_abl_t, bez2_abl_t, _ = self.bezier.bezier_ableitungen(t)

        # Berechnung des Kreuzprodukts
        kreuz_prod = np.cross(bez1_abl_t, bez2_abl_t)

        # Berechnung der Normen
        norm_kreuz_prod = np.linalg.norm(kreuz_prod)
        norm_bez1_abl_t = np.linalg.norm(bez1_abl_t)

        # Berechnung der Krümmung
        kruemmung = norm_kreuz_prod / (norm_bez1_abl_t ** 3)

        return kruemmung

    def torsion_det_formel(self, t):
        '''
        bez1_abl_t, bez2_abl_t, bez3_abl_t = self.bezier.bezier_ableitungen(t)

        # Berechnung des Determinanten der Ableitungen
        det_wert = np.linalg.det(np.vstack((bez1_abl_t, bez2_abl_t, bez3_abl_t)))

        # Berechnung des Kreuzprodukts von bez1_abl_t und bez2_abl_t
        kreuz_prod = np.cross(bez1_abl_t, bez2_abl_t)

        # Betrag des Kreuzprodukts von bez1_abl_t und bez2_abl_t
        kreuz_prod_betrag = np.linalg.norm(kreuz_prod)

        # Überprüfen, ob der Nenner nahe null ist
        if kreuz_prod_betrag < 1e-10:
            return 0.0  # Setze Torsion auf 0, wenn der Nenner zu klein ist

        # Berechnung der Torsion
        torsion = det_wert / (kreuz_prod_betrag ** 2)

        return torsion
        '''

        #
        # Alternative Torsion-Formel zwecks Vergleich zur Determinanten-Formel:
        # \tau = \frac{(\dot{\vec{r}} \times \ddot{\vec{r}}) \cdot \dddot{\vec{r}}}{|\dot{\vec{r}} \times \ddot{\vec{r}}|^2}

        bez1_abl_t, bez2_abl_t, bez3_abl_t = self.bezier.bezier_ableitungen(t)

        # Berechnung des Kreuzprodukts von bez1_abl_t und bez2_abl_t
        kreuz_prod = np.cross(bez1_abl_t, bez2_abl_t)

        # Betrag des Kreuzprodukts von bez1_abl_t und bez2_abl_t
        kreuz_prod_betrag = np.linalg.norm(kreuz_prod)

        # Berechnung des Skalarprodukts von kreuz_prod und bez3_abl_t
        skalar_prod = np.dot(kreuz_prod, bez3_abl_t)

        # Überprüfen, ob der Nenner nahe null ist
        if kreuz_prod_betrag < 1e-10:
            return 0.0  # Setze Torsion auf 0, wenn der Nenner zu klein ist

        # Berechnung der Torsion nach der gegebenen Formel
        torsion = skalar_prod / (kreuz_prod_betrag ** 2)

        return torsion

    def normen_der_ableitungen(self, t):
        bez1_abl_t, bez2_abl_t, bez3_abl_t = self.bezier.bezier_ableitungen(t)

        # Berechnung der Normen
        norm_bez1_abl_t = np.linalg.norm(bez1_abl_t)
        norm_bez2_abl_t = np.linalg.norm(bez2_abl_t)
        norm_bez3_abl_t = np.linalg.norm(bez3_abl_t)

        return norm_bez1_abl_t, norm_bez2_abl_t, norm_bez3_abl_t

    def frenet_serret(self, t):
        bez1_abl_t, bez2_abl_t, bez3_abl_t = self.bezier.bezier_ableitungen(t)

        # Tangentenvektor (T)
        T = bez1_abl_t / np.linalg.norm(bez1_abl_t)

        # Normalenvektor (N)
        kreuz_prod = np.cross(bez2_abl_t, bez1_abl_t)
        N = np.cross(bez1_abl_t, kreuz_prod) / (np.linalg.norm(bez1_abl_t) * np.linalg.norm(kreuz_prod))

        # Binormalenvektor (B)
        B = np.cross(T, N)

        return T, N, B

class BezierAnalyse:
    def __init__(self, stuetzpunkte):
        self.stuetzpunkte = stuetzpunkte
        self.bezier_kurve_berechnung = BezierKurveBerechnung(stuetzpunkte)

    def berechne_fuer_globales_t(self, globales_t, funktion):
        segment_index, lokales_t, _, segment_stuetzpunkte, position = self.bezier_kurve_berechnung.berechne_position_fuer_globales_t(globales_t)

        #Verwendendung der Kontrollpunkte direkt aus bezier_kurve_berechnung
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
    def berechne_alle_werte(self, funktion, anzahl_punkte=2500):
        t_werte = np.linspace(0, 1, anzahl_punkte)
        werte = []

        for t in t_werte:
            wert = self.berechne_fuer_globales_t(t, funktion)
            werte.append(wert)

        return werte

    def berechne_alle_torsionswerte(self, anzahl_punkte=2500):
        return self.berechne_alle_werte(BezierFormeln.torsion_det_formel, anzahl_punkte)

    def berechne_alle_kruemmungswerte(self, anzahl_punkte=2500):
        return self.berechne_alle_werte(BezierFormeln.kruemmung, anzahl_punkte)

    def berechne_alle_normen(self, anzahl_punkte=2500):
        return self.berechne_alle_werte(BezierFormeln.normen_der_ableitungen, anzahl_punkte)


class BezierVisualisierung:
    def __init__(self, bezier_analyse):
        self.bezier_analyse = bezier_analyse

    def plot_torsion(self, t_wert_global, anzahl_punkte=2500):
        werte = self.bezier_analyse.berechne_alle_torsionswerte(anzahl_punkte)
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
            title=f"Torsion der gesamten Bézierkurve (t = {t_wert_global:.2f})<br>Berechneter Wert der Torsion an Stelle t = {t_wert_global:.2f} ist {berechneter_wert:.4f}",
            xaxis=dict(title="Globales t", tickvals=list(np.arange(0, 1.1, 0.1))),
            yaxis_title="Torsion"
        )
        fig.show()

    def plot_kruemmung(self, t_wert_global, anzahl_punkte=2500):
        werte = self.bezier_analyse.berechne_alle_kruemmungswerte(anzahl_punkte)
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
            title=f"Krümmung der gesamten Bézierkurve (t = {t_wert_global:.2f})<br>Berechneter Wert der Krümmung an Stelle t = {t_wert_global:.2f} ist {berechneter_wert:.4f}",
            xaxis=dict(title="Globales t", tickvals=list(np.arange(0, 1.1, 0.1))),
            yaxis_title="Krümmung"
        )
        fig.show()

    def plot_normen(self, t_wert_global, anzahl_punkte=2500):
        werte = self.bezier_analyse.berechne_alle_normen(anzahl_punkte)
        t_werte = np.linspace(0, 1, anzahl_punkte)

        normen_B1_t = [wert[3][0] for wert in werte]
        normen_B2_t = [wert[3][1] for wert in werte]
        normen_B3_t = [wert[3][2] for wert in werte]

        berechneter_wert = self.bezier_analyse.normen_fuer_globales_t(t_wert_global)[3]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_werte, y=normen_B1_t, mode='lines', name='Tempo (1. Ableitung)'))
        fig.add_trace(go.Scatter(x=t_werte, y=normen_B2_t, mode='lines', name='Geschwindigkeit (2. Ableitung)'))
        fig.add_trace(go.Scatter(x=t_werte, y=normen_B3_t, mode='lines', name='Ruck (3. Ableitung)'))
        fig.add_shape(
            type="line",
            x0=t_wert_global,
            x1=t_wert_global,
            y0=0,
            y1=max(max(normen_B1_t), max(normen_B2_t), max(normen_B3_t)),
            line=dict(color="green", width=2)
        )
        fig.update_layout(
            title=f"Normen der Ableitungen der gesamten Bézierkurve (t = {t_wert_global:.2f})<br>Berechneter Wert der Normen an Stelle t = {t_wert_global:.2f} ist {berechneter_wert}",
            xaxis=dict(title="Globales t", tickvals=list(np.arange(0, 1.1, 0.1))),
            yaxis_title="Normen"
        )
        fig.show()

if __name__ == "__main__":
    # Stuetzpunkte definieren
    stuetzpunkte = LeseDatei().trackpunkte_einlesen('_WildeMaus.trk')

    #stuetzpunkte = [(0, 0, 0), (1, 2, 3), (3, 1, 0)]

    # Instanz von BezierKurveBerechnung
    bezier_kurve_berechnung = BezierKurveBerechnung(stuetzpunkte)

    # Kontrollpunkte berechnen
    kontrollpunkte, _, _ = bezier_kurve_berechnung.berechne_kontrollpunkte()

    # Bezierkurve anzeigen
    t_wert_global = 0.5

    grafiken = BezierkurveGrafiken(stuetzpunkte)
    bezier_fig = grafiken.zeichne_wagon_bei_t(t_wert_global)
    bezier_fig.show()

    infos = grafiken.hole_infos_bei_t(t_wert_global)
    print("\n".join(infos))

    # Grafiken generieren
    analyse = BezierAnalyse(stuetzpunkte)
    visualisierung = BezierVisualisierung(analyse)

    visualisierung.plot_torsion(t_wert_global)
    visualisierung.plot_kruemmung(t_wert_global)
    visualisierung.plot_normen(t_wert_global)

    # Torsion für t berechnen
    _, _, _, torsion = analyse.torsion_fuer_globales_t(t_wert_global)
    print(f"\nTorsion bei t = {t_wert_global}: {torsion}")

    # Krümmung für t berechnen
    _, _, _, kruemmung = analyse.kruemmung_fuer_globales_t(t_wert_global)
    print(f"Krümmung bei t = {t_wert_global}: {kruemmung}")

    # Normen für t berechnen
    _, _, _, normen = analyse.normen_fuer_globales_t(t_wert_global)
    print(f"Normen bei t = {t_wert_global}: {normen}")